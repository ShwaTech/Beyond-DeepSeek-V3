from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


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
