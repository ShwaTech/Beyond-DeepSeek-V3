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