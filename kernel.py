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