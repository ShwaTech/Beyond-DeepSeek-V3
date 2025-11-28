from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


# =======================================
# Activation Quantization Kernel
# =======================================
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, scale_fmt: tl.constexpr):
    """
    A Triton kernel for quantizing activation values block-wise.

    This kernel performs:
        1. Load a block of input activations from memory
        2. Compute block-wise absolute max
        3. Compute quantization scale 's'
        4. Quantize the values into 8-bit range (ue8m0 / fake FP8)
        5. Store quantized values + scale

    This enables **FP8 activation quantization**, which DeepSeek-V3 uses heavily
    to reduce memory bandwidth and improve throughput.

    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
        scale_fmt (tl.constexpr): Format of scale ("ue8m0" means power-of-two scaling).

    Returns:
        None
    """
    # ---------------------------------------------------------
    # 1. Identify which program instance we are (like CUDA blockIdx.x)
    # ---------------------------------------------------------
    pid = tl.program_id(axis=0)
    # Example:
    #   pid = 0 → process elements 0 .. BLOCK_SIZE-1
    #   pid = 1 → process elements BLOCK_SIZE .. 2*BLOCK_SIZE-1
    # This allows us to tile the tensor into blocks.

    # ---------------------------------------------------------
    # 2. Compute global memory offsets for this block
    #    offs is a vector: [pid*BS + 0, pid*BS + 1, ..., pid*BS + BS-1]
    # ---------------------------------------------------------
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ---------------------------------------------------------
    # 3. Load a block of activations from global memory
    #    Convert to FP32 for stable math
    # ---------------------------------------------------------
    x = tl.load(x_ptr + offs).to(tl.float32)

    # ---------------------------------------------------------
    # 4. Find the absolute maximum value inside the block (reduction)
    #    This is needed for symmetric quantization
    # ---------------------------------------------------------
    amax = tl.max(tl.abs(x))   # this is a single FP32 scalar

    # ---------------------------------------------------------
    # 5. Clamp amax to prevent small numbers causing huge scale
    # ---------------------------------------------------------
    amax = tl.maximum(amax, 1e-4)

    # ---------------------------------------------------------
    # 6. Compute quantization scale
    #
    #    DeepSeek uses FP8-like formats.
    #    The max representable range for 8-bit unsigned is ~448.
    #
    #    So scale = max_abs / 448
    # ---------------------------------------------------------
    s = amax / 448.0

    # ---------------------------------------------------------
    # 7. If scale format == "ue8m0", force scale to nearest power-of-two
    #
    #    This enables:
    #        • faster multiply/divide (bitshift)
    #        • hardware friendliness
    #        • predictable ranges (like FP8)
    #
    # ---------------------------------------------------------
    if scale_fmt == "ue8m0":
        exp = tl.math.ceil(tl.math.log2(s))   # exponent of scale
        s = tl.math.exp2(exp)                 # 2^exp — power-of-two scale

    # ---------------------------------------------------------
    # 8. Quantize:
    #
    #        y = x / s
    #
    #    This maps FP32 → something like FP8 range
    # ---------------------------------------------------------
    y = x / s

    # ---------------------------------------------------------
    # 9. Cast to the dtype of the output buffer (usually uint8 or int8)
    # ---------------------------------------------------------
    y = y.to(y_ptr.dtype.element_ty)

    # ---------------------------------------------------------
    # 10. Store quantized block
    # ---------------------------------------------------------
    tl.store(y_ptr + offs, y)

    # ---------------------------------------------------------
    # 11. Store per-block scale for later dequantization
    #     (only 1 scale per block — very efficient!)
    # ---------------------------------------------------------
    tl.store(s_ptr + pid, s)



# =======================================
# Activation Quantization Function
# =======================================
def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    This is the high-level Python wrapper that:
        1. Validates the input tensor.
        2. Allocates output tensors (quantized data + scales).
        3. Launches the Triton GPU kernel.
        4. Returns quantized FP8 values and per-block scales.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    
    # ----------------------------------------------------------------------
    # 1. Safety checks — ensure memory layout matches Triton kernel's expectations
    # ----------------------------------------------------------------------
    assert x.is_contiguous(), "Input tensor must be contiguous"
    # Triton accesses x by simple pointer arithmetic, so it must be contiguous.

    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    # The last dimension is divided into BLOCKS of size = block_size.
    # Example: shape (..., 4096), block_size=128 → 4096/128 = 32 blocks.

    # ----------------------------------------------------------------------
    # 2. Allocate output tensor for quantized results
    # ----------------------------------------------------------------------
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    #
    # Why float8_e4m3fn?
    #   • This is PyTorch’s FP8 format (E4M3)
    #   • It matches DeepSeek’s activation quantization format
    #   • Stored as 8 bits but treated as float on CUDA kernels

    # ----------------------------------------------------------------------
    # 3. Allocate the scale tensor
    # ----------------------------------------------------------------------
    s = x.new_empty(
        *x.size()[:-1],             # keep all dims except the last
        x.size(-1) // block_size,   # one scale per block
        dtype=torch.float32
    )
    #
    # Example:
    #   x shape : [32, 4096]
    #   blocks  : 4096 / 128 = 32
    #   s shape : [32, 32]
    #
    # This stores one FP32 scale per 128-element block.

    # ----------------------------------------------------------------------
    # 4. Define launch grid for Triton kernel
    # ----------------------------------------------------------------------
    grid = lambda meta: (
        triton.cdiv(x.numel(), meta['BLOCK_SIZE']),
    )
    # grid is the number of “program instances.”
    # Triton kernel processes BLOCK_SIZE elements per program.
    # num_programs = ceil(total_elements / BLOCK_SIZE)
    # This automatically spreads work across GPU.

    # ----------------------------------------------------------------------
    # 5. Launch the Triton kernel
    # ----------------------------------------------------------------------
    act_quant_kernel[grid](
        x,      # input activations
        y,      # output quantized activations
        s,      # output scale tensor
        BLOCK_SIZE=block_size,
        scale_fmt=scale_fmt,
    )
    # Kernel performs:
    #   • block load
    #   • absmax reduction
    #   • scale computation
    #   • optional power-of-two scaling adjustment
    #   • FP8 quantization
    #   • writing quantized output + scale

    # ----------------------------------------------------------------------
    # 6. Return quantized tensor and scales
    # ----------------------------------------------------------------------
    return y, s



# =======================================
# Weight Dequantization Kernel
# =======================================
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    # ----------------------------------------------------------------------
    # 1. Identify Which Tile This Kernel Instance Should Process
    # ----------------------------------------------------------------------
    # Triton launches many "programs" (like CUDA thread blocks).
    # Each program handles one tile of the matrix.
    pid_m = tl.program_id(axis=0)    # tile index in the row dimension
    pid_n = tl.program_id(axis=1)    # tile index in the column dimension

    # ----------------------------------------------------------------------
    # 2. Number of Column Tiles (for locating scale index)
    # ----------------------------------------------------------------------
    # N may not divide BLOCK_SIZE exactly. cdiv = ceil(N / BLOCK_SIZE).
    n_col_blocks = tl.cdiv(N, BLOCK_SIZE)

    # ----------------------------------------------------------------------
    # 3. Compute Row and Column Offsets for This Tile
    # ----------------------------------------------------------------------
    # offs_m = [row0, row1, row2, ..., row(BLOCK_SIZE-1)]
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # offs_n = [col0, col1, col2, ..., col(BLOCK_SIZE-1)]
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Produce a full (BLOCK_SIZE × BLOCK_SIZE) grid of linear indices:
    # linear_index = row * N + col
    # Broadcasting:
    #   offs_m[:, None] shape = (BLOCK_SIZE, 1)
    #   offs_n[None, :] shape = (1, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]

    # ----------------------------------------------------------------------
    # 4. Mask for Out-of-Bounds (Edges of Matrix)
    # ----------------------------------------------------------------------
    # Ensures loads and stores only occur where indices are valid.
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # ----------------------------------------------------------------------
    # 5. Load Quantized Weights for This Tile
    # ----------------------------------------------------------------------
    # Read BLOCK_SIZE×BLOCK_SIZE values from x_ptr.
    # Unsafe locations masked → zero-filled instead of crashing.
    x_q = tl.load(x_ptr + offs, mask=mask)

    # Convert quantized values (float8/int8) → float32 for math.
    x_fp32 = x_q.to(tl.float32)

    # ----------------------------------------------------------------------
    # 6. Load Scale Factor for This Tile
    # ----------------------------------------------------------------------
    # One scale per tile:
    #   block_index = pid_m * n_col_blocks + pid_n
    s = tl.load(s_ptr + pid_m * n_col_blocks + pid_n)

    # ----------------------------------------------------------------------
    # 7. Perform Dequantization
    # ----------------------------------------------------------------------
    # W_fp32 = W_quantized * scale
    y = x_fp32 * s

    # ----------------------------------------------------------------------
    # 8. Store Result into Output Buffer
    # ----------------------------------------------------------------------
    # Mask ensures we only write valid positions.
    tl.store(y_ptr + offs, y, mask=mask)



# =======================================
# Weight Dequantization Function
# =======================================
def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    # ----------------------------------------------------------------------
    # 1. Validate Tensor Layout (Important for Triton Memory Access)
    # ----------------------------------------------------------------------
    assert x.is_contiguous() and s.is_contiguous(), \
        "Input tensors must be contiguous for efficient GPU loads/stores."

    # Ensure both inputs are rank-2 matrices
    assert x.dim() == 2 and s.dim() == 2, \
        "x and s must be 2D matrices (M×N and M/block × N/block)."

    # ----------------------------------------------------------------------
    # 2. Extract Matrix Shape
    # ----------------------------------------------------------------------
    M, N = x.size()      # M rows, N columns

    # ----------------------------------------------------------------------
    # 3. Allocate Output Matrix for Dequantized FP32 Weights
    # ----------------------------------------------------------------------
    # dtype=torch.get_default_dtype() ensures FP32 or BF16 depending on PyTorch config
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    # ----------------------------------------------------------------------
    # 4. Define Triton Kernel Launch Grid
    # ----------------------------------------------------------------------
    # Triton launches a 2D grid:
    #   grid = (#row_tiles, #column_tiles)
    #
    # Each tile = block_size × block_size region of the weight matrix.
    #
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE']),  # number of row tiles
        triton.cdiv(N, meta['BLOCK_SIZE']),  # number of column tiles
    )

    # ----------------------------------------------------------------------
    # 5. Launch The Dequantization Kernel
    # ----------------------------------------------------------------------
    # This calls the compiled Triton kernel:
    #
    #   weight_dequant_kernel(x, s, y, M, N, BLOCK_SIZE=block_size)
    #
    # The kernel:
    #   - loads a tile of quantized weights
    #   - loads the proper blockwise scale
    #   - multiplies (in FP32)
    #   - writes into y
    #
    weight_dequant_kernel[grid](
        x,      # pointer to quantized matrix
        s,      # pointer to scale matrix
        y,      # output FP32 buffer
        M, N,   # matrix dimensions
        BLOCK_SIZE=block_size
    )

    # ----------------------------------------------------------------------
    # 6. Return Full Dequantized Matrix
    # ----------------------------------------------------------------------
    return y



# ------------------------------------------------------------------------------
# fp8_gemm_configs:
# A list of candidate kernel configurations for FP8 GEMM autotuning.
#
# DeepSeek creates MANY configurations varying:
#   - BLOCK_SIZE_M (tile height)
#   - BLOCK_SIZE_N (tile width)
#   - BLOCK_SIZE_K (inner dimension tile size)
#   - num_stages   (pipeline depth for software prefetching)
#   - num_warps    (# of warps per Triton program instance)
#
# Triton’s autotuner will benchmark these to pick the fastest kernel on your GPU.
# ------------------------------------------------------------------------------
fp8_gemm_configs = [
    Config(
        # Kernel tile parameters:
        {
            'BLOCK_SIZE_M': block_m,  # How many output rows this kernel tile computes
            'BLOCK_SIZE_N': block_n,  # How many output columns this tile computes
            'BLOCK_SIZE_K': 128       # Tile size along the reduction dimension K (fixed)
        },
        num_stages=num_stages,  # Pipeline depth: 3..6 async load stages
        num_warps=8             # Number of warps executing the kernel (parallelism)
    )

    # --------------------------------------------------------------------------
    # Cartesian product:
    #   block_m in [16, 32, 64]
    #   block_n in [32, 64, 128]
    #   num_stages in [3, 4, 5, 6]
    #
    # This produces 3 × 3 × 4 = 36 unique kernel configurations.
    # --------------------------------------------------------------------------
    for block_m in [16, 32, 64]       # Tile height options
    for block_n in [32, 64, 128]      # Tile width options
    for num_stages in [3, 4, 5, 6]    # Prefetch pipeline depth (latency hiding)
]



# =======================================
# FP8 GEMM Kernel
# =======================================
@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    # ------------------------------------------------------------
    # 1) tile coordinates handled by this program instance
    # ------------------------------------------------------------
    pid_m = tl.program_id(axis=0)   # which tile row
    pid_n = tl.program_id(axis=1)   # which tile column

    # number of K-chunks to iterate (ceil)
    num_k_chunks = tl.cdiv(K, BLOCK_SIZE_K)

    # ------------------------------------------------------------
    # 2) compute coordinates (logical indices) for rows/cols in this tile
    # ------------------------------------------------------------
    # NOTE: we avoid using modulo (%) for indexing into memory; modulo can *wrap*
    # and introduce incorrect repeated reads when M or N are not multiples of block size.
    # Instead, compute the base indices and use masks for bounds checking.
    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # shape: (BLOCK_SIZE_M,)
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # shape: (BLOCK_SIZE_N,)

    # K offsets in a single chunk (shared across the inner loop)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)                            # shape: (BLOCK_SIZE_K,)

    # linear offsets to memory for initial K-chunk:
    # a_base_ptrs shape -> (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_ptrs = a_ptr + row_offsets[:, None] * K + k_offsets[None, :]
    # b_base_ptrs shape -> (BLOCK_SIZE_M? no, we'll compute shape for load below)
    # for B we want (BLOCK_SIZE_K, BLOCK_SIZE_N) when we load
    b_ptrs = b_ptr + k_offsets[:, None] * N + col_offsets[None, :]

    # ------------------------------------------------------------
    # 3) prepare scale indexing (EXPLICIT and SAFE)
    # ------------------------------------------------------------
    # We expect a_s_ptr to contain one scale per (row_tile, k_chunk)
    # and b_s_ptr to contain one scale per (k_chunk, col_tile).
    #
    # Layout assumptions (choose the layout you used at quant time):
    #   a_s layout shape: [num_row_tiles, num_k_chunks]  (index = row_tile * num_k_chunks + k_chunk)
    #   b_s layout shape: [num_k_chunks, num_col_tiles]  (index = k_chunk * num_col_tiles + col_tile)
    #
    # Compute identifiers for this tile:
    row_tile_id = pid_m                 # 0 .. ceil(M/BM)-1
    col_tile_id = pid_n                 # 0 .. ceil(N/BN)-1
    num_col_tiles = tl.cdiv(N, BLOCK_SIZE_N)

    # We'll create pointers for a_s and b_s that we increment inside the loop.
    # a_s_index_base = row_tile_id * num_k_chunks
    a_s_ptrs = a_s_ptr + row_tile_id * num_k_chunks
    # b_s_index_base = col_tile_id (but b_s stored per k_chunk × col_tile)
    # so b scale index for (k_chunk, col_tile) = k_chunk * num_col_tiles + col_tile_id
    # We'll step b_s by num_col_tiles each iteration.
    b_s_ptrs_base = b_s_ptr + col_tile_id

    # ------------------------------------------------------------
    # 4) accumulator (FP32)
    # ------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ------------------------------------------------------------
    # 5) K-loop: iterate over each reduction chunk
    # ------------------------------------------------------------
    # Each iteration processes BLOCK_SIZE_K elements of the K dimension.
    for k_idx in range(num_k_chunks):
        # --- load A tile for this k-chunk:
        # mask for A load: True where (row_offset < M) and (k_offset + k_idx*BLOCK_SIZE_K < K)
        a_k_remaining = K - k_idx * BLOCK_SIZE_K
        a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < a_k_remaining)

        # load quantized A block and cast to float32
        a_q = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)   # shape (BM, BK)

        # --- load B tile for this k-chunk:
        b_k_remaining = K - k_idx * BLOCK_SIZE_K
        b_mask = (k_offsets[:, None] < b_k_remaining) & (col_offsets[None, :] < N)
        b_q = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)   # shape (BK, BN)

        # --- load scales for this k-chunk:
        # a_s_ptrs points to [row_tile_id, k_idx] (layout row_tile major)
        a_s = tl.load(a_s_ptrs + k_idx)   # scalar or vector length 1
        # b_s_ptrs_base points to [k_idx, col_tile_id] if we step by num_col_tiles
        b_s = tl.load(b_s_ptrs_base + k_idx * num_col_tiles)   # scalar

        # NOTE: a_s and b_s are scalars for the whole block. If you stored one scale
        # per row element or per column element inside the block, you'll need to adjust shapes.

        # --- Accumulate:
        # dot(a_q (BM×BK), b_q (BK×BN)) -> (BM×BN)
        # then scale by outer product a_s[:, None] * b_s[None, :]
        # since a_s and b_s are scalars here, this just multiplies the dot by a_s*b_s
        acc += tl.dot(a_q, b_q) * (a_s * b_s)

        # --- advance pointers to the next K-chunk
        a_ptrs += BLOCK_SIZE_K                # move the K-window for A
        b_ptrs += BLOCK_SIZE_K                # move the K-window for B (due to how we built b_ptrs)
        # a_s_ptrs and b_s_ptrs_base are indexed by k_idx so we don't need to physically increment them

    # ------------------------------------------------------------
    # 6) write accumulator to C (with bounds mask)
    # ------------------------------------------------------------
    # compute the final linear pointers for C tile
    write_row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    write_col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + write_row_offsets[:, None] * N + write_col_offsets[None, :]

    write_mask = (write_row_offsets[:, None] < M) & (write_col_offsets[None, :] < N)

    # cast accumulator to output element type and store with mask
    c_out = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c_out, mask=write_mask)