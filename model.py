import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"



@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.



class ParallelEmbedding(nn.Module):
    """
    Parallelized Embedding Layer — used in DeepSeek-V3 for **vocabulary model parallelism**.

    This layer shards the embedding matrix across `world_size` GPUs.
    Each GPU holds only a *slice* of the full embedding table.
    During the forward pass:
        - Each GPU locally embeds the tokens that belong to its vocabulary slice.
        - Tokens that do *not* belong to the slice are masked to zero.
        - The final embeddings from all GPUs are summed using all_reduce,
          giving the same result as if each GPU contained the *full* embedding table.

    Why?
        ✔ Saves memory — each GPU stores only vocab_size / world_size embeddings.
        ✔ Scales to multi-10B vocab embeddings.
        ✔ Enables distributed training with large vocabularies.

    Args:
        vocab_size (int):
            Total vocabulary size (shared across all GPUs).
        dim (int):
            Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        # -------------------------------------------------------------
        # DeepSeek-style vocab parallelism requires vocab_size to be
        # divisible by the number of GPUs in the tensor-parallel group.
        # -------------------------------------------------------------
        assert vocab_size % world_size == 0, \
            f"Vocabulary size must be divisible by world size (world_size={world_size})"

        # -------------------------------------------------------------
        # How many vocabulary tokens THIS GPU is responsible for?

        # Split the embedding matrix across GPUs:
        #
        # GPU0 → tokens [0, 25k)
        # GPU1 → tokens [25k, 50k)
        # GPU2 → tokens [50k, 75k)
        # GPU3 → tokens [75k, 100k)
        #
        # Example:
        #   vocab_size = 100k, world_size = 4
        #   part_vocab_size = 25k tokens per GPU
        # -------------------------------------------------------------
        self.part_vocab_size = vocab_size // world_size

        # -------------------------------------------------------------
        # Determine this GPU's vocabulary range.
        #
        # If rank=1 and part_vocab_size=25k:
        #   vocab_start_idx = 25k
        #   vocab_end_idx   = 50k
        #
        # Only tokens in this range belong to THIS GPU.
        # -------------------------------------------------------------
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx   = self.vocab_start_idx + self.part_vocab_size

        # -------------------------------------------------------------
        # Allocate this GPU's slice of the embedding matrix.
        #
        # Shape: [part_vocab_size, dim]
        #
        # Note: This is a PARAMETER (learnable).
        # -------------------------------------------------------------
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distributed forward pass for vocab-parallel embeddings.

        Args:
            x (torch.Tensor):
                Tensor containing token IDs in range [0, vocab_size).

        Returns:
            torch.Tensor:
                Embedded token representations, identical to those produced
                by a full embedding table — but computed in parallel.

        Magical Behavior (DeepSeek-V3 style):
            - Each GPU embeds *only the tokens it owns*.
            - Tokens belonging to other GPUs are masked and zeroed.
            - all_reduce then gathers contributions from all GPUs,
                ensuring every token gets its correct full embedding vector.
        """
        # -----------------------------------------------------------------
        # Case: world_size = 1 → no model parallelism needed.
        # -----------------------------------------------------------------
        if world_size > 1:

            # -------------------------------------------------------------
            # Determine which tokens belong to THIS GPU.
            #
            # mask == True → for tokens NOT in our local vocabulary slice.
            #
            # Example:
            #   GPU 1 owns vocab IDs [25k, 50k)
            #
            #   If x = [10k, 27k, 80k], mask = [True, False, True]
            # -------------------------------------------------------------
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)

            # -------------------------------------------------------------
            # Adjust token indices from global to local vocabulary space.
            #
            # If GPU owns range [25k, 50k):
            #   global ID 25,000 → local ID 0
            #   global ID 25,001 → local ID 1
            #
            # This allows F.embedding() to access the *local* weight matrix.
            # -------------------------------------------------------------
            x = x - self.vocab_start_idx

            # -------------------------------------------------------------
            # For tokens not owned by this GPU:
            #   - Set index to 0 (a dummy index)
            #   - These will be zeroed after embedding anyway
            #
            # Why set to 0?
            #   Because F.embedding() does NOT allow invalid indices.
            # -------------------------------------------------------------
            x[mask] = 0

        # -----------------------------------------------------------------
        # Perform the actual embedding lookup.
        # For GPUs in parallel mode, this embeds only locally-owned tokens.
        #
        # Output shape: [*input_shape, dim]
        # -----------------------------------------------------------------
        y = F.embedding(x, self.weight)

        if world_size > 1:
            # -------------------------------------------------------------
            # Zero out embeddings for tokens that did NOT belong to this GPU.
            #
            # y[mask] currently contains garbage (embedding of dummy index 0)
            # since earlier we replaced out-of-range tokens with index 0.
            #
            # Set them to ZERO to avoid incorrect contributions.
            # -------------------------------------------------------------
            y[mask] = 0

            # -------------------------------------------------------------
            # all_reduce SUM:
            #
            # Each GPU contributes:
            #   - embeddings for tokens it owns
            #   - zeros for tokens it doesn't own
            #
            # Summing across GPUs reconstructs the FULL embedding table.
            #
            # This is the CORE IDEA of parallel embeddings.
            # -------------------------------------------------------------
            dist.all_reduce(y)

        return y



def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale_fmt: Optional[str] = None
) -> torch.Tensor:
    """
    Apply a linear transformation: y = x * W^T + b.

    This function automatically selects the appropriate GEMM backend depending
    on tensor formats (e.g., FP8, BF16) and quantization state.

    Args:
        x (torch.Tensor):
            Input activation tensor.
        weight (torch.Tensor):
            Weight matrix, possibly quantized (e.g., FP8 with element_size() == 1).
        bias (Optional[torch.Tensor], optional):
            Optional bias to add to the output. Defaults to None.
        scale_fmt (Optional[str], optional):
            Optional scaling format used for FP8 activation quantization.

    Returns:
        torch.Tensor:
            Output tensor after applying linear transformation using the appropriate
            GEMM implementation.

    Execution Paths:
        1. **Unquantized weights (FP16/BF16/FP32)**
                → Uses standard `torch.nn.functional.linear`.

        2. **Quantized weights + BF16 GEMM**
                → Dequantizes weights, then calls standard BF16 linear.

        3. **Quantized weights + FP8 GEMM**
                → Quantizes activations using `act_quant`
                → Computes matrix multiplication using `fp8_gemm`
                → Adds bias if provided
    """
    # Case 1: Weight not quantized (standard FP16/FP32/BF16)
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)

    # Case 2: Quantized weights but BF16 GEMM is requested
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)

    # Case 3: FP8 GEMM path (quantized weights + fp8_gemm)
    else:
        # Quantize activations (x → FP8)
        x_fp8, x_scale = act_quant(x, block_size, scale_fmt)

        # GEMM using FP8 inputs and scales
        y = fp8_gemm(x_fp8, x_scale, weight, weight.scale)

        # Add bias (if provided)
        if bias is not None:
            y += bias

        return y



class Linear(nn.Module):
    """
    Custom linear layer supporting FP8-quantized weights, optional bias,
    and automatic dispatch to the appropriate GEMM implementation.

    Args:
        in_features (int):
            Number of input features.
        out_features (int):
            Number of output features.
        bias (bool, optional):
            Whether to include a learnable bias. Defaults to False.
        dtype (optional):
            Data type for storing weights (e.g., torch.bfloat16 or FP8).
            If None, defaults to `Linear.dtype`.

    Notes:
        - If `dtype` results in 1-byte storage (FP8), a block-wise scale tensor
            is created and stored as `weight.scale`.
        - For non-quantized types (BF16/FP16/FP32), no scale parameter is created.
        - The forward pass delegates computation to the `linear()` function,
            which chooses FP32, BF16, or FP8 GEMM depending on tensor formats.
    """

    # Default datatype for the layer unless explicitly overridden
    dtype = torch.bfloat16

    # Optional activation scaling format for FP8 quantization
    scale_fmt: Optional[str] = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight (may be FP8 or BF16/FP16/FP32)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype or Linear.dtype)
        )

        # If weight is quantized (element_size() == 1 → FP8)
        if self.weight.element_size() == 1:
            scale_out = (out_features + block_size - 1) // block_size
            scale_in = (in_features + block_size - 1) // block_size

            # Block-wise scale for FP8 weight matrix
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out, scale_in, dtype=torch.float32)
            )
        else:
            # No scale needed for non-quantized weights
            self.register_parameter("scale", None)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation y = xW^T + b,
        using FP8, BF16, or standard FP32 computations
        depending on the layer’s weight format.

        Args:
            x (torch.Tensor):
                Input activation tensor.

        Returns:
            torch.Tensor:
                Output tensor after linear transformation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)



class ColumnParallelLinear(Linear):
    """
    Column-parallel linear layer used in tensor model parallelism.
    Splits the *output features* across distributed processes (ranks), so each
    rank holds a slice of the full weight matrix along the output dimension.

    This layer implements:
        W = [W_0, W_1, ..., W_{world_size-1}]  (split by columns)

    Meaning each rank stores:
        W_i ∈ R[out_features/world_size, in_features]

    And computes its partial output:
        y_i = x @ W_i^T + b_i

    These partial results can either:
        - be returned directly (for intermediate transformer blocks), or
        - be reduced/concatenated later depending on model architecture.

    Args:
        in_features (int):
            Number of input features.
        out_features (int):
            Total number of output features before splitting.
        bias (bool):
            Whether to include a bias term. Defaults to False.
        dtype (optional):
            Data type for storing weights (FP8/BF16/etc.).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None
    ):
        # Ensure output features are evenly divisible across tensor-parallel ranks
        assert (
            out_features % world_size == 0
        ), f"Output features must be divisible by world_size={world_size}"

        # Number of output units handled by THIS rank only
        self.part_out_features = out_features // world_size

        # Initialize parent Linear module with the sliced output dimension
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the column-parallel forward pass.

        Each rank performs:
            y_i = x @ W_i^T + b_i

        - No all_reduce is performed here.
        - The caller is responsible for gathering results if full output is needed.
        - This matches Megatron-LM style tensor parallelism, also adopted by DeepSeek-V3.

        Args:
            x (torch.Tensor):
                Input tensor of shape [..., in_features].

        Returns:
            torch.Tensor:
                Partial output tensor of shape [..., out_features/world_size].
                (Each rank returns its local column-slice of the output.)
        """

        # Uses the global 'linear' function, which dispatches to:
        # - FP32/BF16 F.linear for normal weights
        # - FP8 optimized GEMM path for quantized weights
        # - Block-wise scaling logic when needed (FP8 acts)
        y = linear(x, self.weight, self.bias)

        return y



class RowParallelLinear(Linear):
    """
    Row-parallel linear layer used in tensor model parallelism.
    Splits the *input features* (rows of the weight matrix) across distributed
    processes/ranks. Each rank stores a block of the weight matrix along the
    input dimension and computes a *partial contribution* to the final output.

    Weight layout across ranks:
        Full weight:   W ∈ R[out_features, in_features]
        Sharded as:    W_i ∈ R[out_features, in_features / world_size]
        Where:
            W = concat([W_0, W_1, ..., W_{N-1}], dim=1)

    Each rank receives its own input slice:
        x_i ∈ R[..., in_features/world_size]

    And computes its partial output:
        y_i = x_i @ W_i^T

    Since each rank contributes *part of every output unit*, an all_reduce is
    required to aggregate partial sums:
        y = sum_i y_i

    This matches the Megatron-LM row-parallel design, and is used in DeepSeek-V3
    for attention output projections and MLP second GEMM layers.

    Args:
        in_features (int):
            Total number of input features before sharding.
        out_features (int):
            Number of output features (replicated across ranks).
        bias (bool):
            Whether to include a bias term. Defaults to False.
        dtype (optional):
            Weight data type (BF16/FP8/etc.).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None
    ):
        # Ensure input features evenly split across tensor-parallel ranks
        assert (
            in_features % world_size == 0
        ), f"Input features must be divisible by world_size={world_size}"

        # Slice of input features handled by THIS rank only
        self.part_in_features = in_features // world_size

        # Initialize parent Linear with sharded input dimension
        # (Each rank stores W_i with shape [out_features, part_in_features])
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the row-parallel forward pass.

        Inputs:
            x should already be sharded across ranks:
                x_i ∈ R[..., in_features/world_size]

        Local computation:
            y_i = x_i @ W_i^T   (partial output)

        Distributed aggregation:
            y = sum_i y_i       (via all_reduce)

        This is necessary because each rank contributes a partial sum to
        *all* output features.

        Bias is added after the reduction since bias is replicated across ranks.

        Args:
            x (torch.Tensor):
                Local input slice for this rank.

        Returns:
            torch.Tensor:
                Full output tensor after summing all partial results.
                Shape: [..., out_features]
        """

        # Compute local partial output using the global 'linear' implementation
        # (handles FP8 dequant, BF16 GEMM, or default F.linear)
        y = linear(x, self.weight)

        # Aggregate partial outputs across all tensor-parallel ranks
        if world_size > 1:
            dist.all_reduce(y)

        # Bias is applied after reduction (bias is identical on all ranks)
        if self.bias is not None:
            y += self.bias

        return y



class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm is a simplified normalization method used in modern large-scale
    architectures (DeepSeek-V3, Llama, Falcon) because it is:
        - Computationally cheaper than LayerNorm
        - More numerically stable at very large hidden sizes
        - Better suited for FP8 and BF16 mixed-precision compute
        - Friendlier to distributed execution pipelines

    Unlike LayerNorm, RMSNorm does *not* subtract the mean of activations.
    Instead, it normalizes only by the root-mean-square (RMS):

        y = x * weight / sqrt(mean(x^2) + eps)

    This removes the need for both:
        - Mean-centering (x - mean)
        - A learned bias parameter

    Making it significantly simpler and faster, especially in highly parallel
    environments like DeepSeek-V3.

    Args:
        dim (int):
            The last-dimension size of the input tensor. RMSNorm normalizes
            across this dimension.

        eps (float):
            Small epsilon added to the denominator for numerical stability.
            Prevents division-by-zero in FP8/BF16 execution.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        # Store configuration
        self.dim = dim
        self.eps = eps

        # RMSNorm has only one learned parameter: scale (γ)
        # This is applied elementwise after normalization.
        #
        # No bias term is used because RMSNorm does not shift activations.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input.

        PyTorch provides F.rms_norm() that directly implements:
            norm = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
            y = norm * weight

        Input:
            x : Tensor[..., dim]
                Arbitrary leading dimensions, normalization applies to last dim.

        Returns:
            Tensor[..., dim]
                RMS-normalized output, scaled by learned weight.
        """

        # Delegate to the optimized PyTorch fused implementation.
        # This is important for DeepSeek-V3 because:
        #   - It uses vectorized RMS computation
        #   - Runs efficiently under FP8/BF16 precision
        #   - Works well with fused kernels and CUDA graph capture
        return F.rms_norm(
            x,
            (self.dim,),   # Normalize across this dimension
            self.weight,   # Learned scale parameter γ
            self.eps       # Numerical stability
        )



def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precompute the complex-valued rotary frequencies (cis = cos + i·sin) used in
    DeepSeek-V3’s improved RoPE mechanism.

    DeepSeek-V3 uses a *corrected*, *multi-scale*, and *extrapolation-safe*
    version of Rotary Positional Embeddings (RoPE). This enhanced formulation
    fixes the classic problem that RoPE breaks when sequence length exceeds the
    length used during training.

    This function performs four key steps:

    1) Build base inverse-frequency RoPE spectrum:
        freq[i] = 1 / base^(i/dim)

    2) If the current sequence length exceeds the training-time limit,
        compute correction bands based on:
            - β_fast : rotations that drift fastest (high-frequency heads)
            - β_slow : rotations that drift slowly (low-frequency heads)

    3) Apply DeepSeek-V3’s “Smooth RoPE Scaling”:
        - Uses a *ramping mask* that linearly blends between:
            • original frequencies (trained range)
            • scaled frequencies (extrapolation range)

    4) Convert raw angular frequencies into complex exponentials:
        cis(θ) = cos(θ) + i⋅sin(θ)

        This lets the attention rotate Q and K vectors using fast,
        element-wise complex multiplication.

    Returns:
        torch.Tensor of shape (seqlen, dim/2)
            Complex cis values used for rotating (q, k) vectors during attention.
    """

    # Dimension of each head affected by RoPE (only half gets sin/cos pairs)
    dim = args.qk_rope_head_dim

    # Maximum sequence length we want to support at runtime
    seqlen = args.max_seq_len

    # DeepSeek-V3 multi-scale extrapolation parameters
    beta_fast = args.beta_fast     # high-frequency correction threshold
    beta_slow = args.beta_slow     # low-frequency correction threshold

    # Classical RoPE hyperparameters
    base = args.rope_theta         # usually 10_000 but tunable
    factor = args.rope_factor      # scaling factor used during extrapolation

    # -------------------------------------------------------------------------
    # Helper: Compute which RoPE dimension corresponds to a given number of
    #         complete positional rotations over the training sequence length.
    #
    # The RoPE frequency grows exponentially with dimension index:
    #     freq[k] ≈ base^(-k/dim)
    #
    # The number of rotations over the original training window is:
    #     rotations = seq_len * freq
    #
    # We solve for k (dimension index) such that rotations == desired_rot
    # -------------------------------------------------------------------------
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Solve for the RoPE dimension index 'k' that produces a target number
        of rotations over the training sequence window.

        This is derived from:
            num_rot = seq_len * base^(-k/dim)
            -> solve for k
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    # -------------------------------------------------------------------------
    # Helper: Compute the low/high dimension indices whose frequencies lie
    #         between β_fast and β_slow rotations. These boundaries define the
    #         "correction band" where smoothing is applied.
    # -------------------------------------------------------------------------
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Find frequency band indices that require extrapolation correction.
        Returns (low_idx, high_idx) clamped to valid range.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    # -------------------------------------------------------------------------
    # Helper: Compute a smooth ramping mask from 0 → 1 between two dimensions.
    #
    # Used to softly blend RoPE frequencies:
    #     blended = corrected_freq * (1 - smooth) + original_freq * smooth
    #
    # This avoids sudden discontinuities in rotation rates.
    # -------------------------------------------------------------------------
    def linear_ramp_factor(min, max, dim):
        """
        Linearly interpolate from 0 to 1 between indices [min, max].
        Clamped outside the range.

        Example:
            min=20, max=50, dim=100 produces a 100-length vector where:
                [0..20] → 0
                [20..50] → linear ramp 0 → 1
                [50..] → 1
        """
        if min == max:
            # Prevent division by zero
            max += 0.001

        linear = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear, 0, 1)

    # =========================================================================
    # Step 1 — Compute base inverse frequencies (standard RoPE)
    # =========================================================================
    # RoPE uses frequencies 1 / base^(i/dim) applied to (sin, cos) pairs.
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # =========================================================================
    # Step 2 — If we are *extrapolating* beyond the trained seq length,
    #          apply DeepSeek-V3 RoPE frequency corrections.
    # =========================================================================
    if seqlen > args.original_seq_len:
        # Determine the correction band for extrapolation safety
        low, high = find_correction_range(
            beta_fast,            # high-frequency behavior (fast drift)
            beta_slow,            # low-frequency behavior (slow drift)
            dim,
            base,
            args.original_seq_len
        )

        # Smooth transition mask between corrected and original frequencies
        # Shape: (dim/2,)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)

        # Apply DeepSeek-V3’s multi-scale RoPE correction:
        #   - Low-frequency dims receive small correction
        #   - High-frequency dims remain original
        freqs = (
            freqs / factor * (1 - smooth) +  # corrected (scaled) frequencies
            freqs * smooth                   # original frequencies
        )

    # =========================================================================
    # Step 3 — Build time-indexed angles θ = t * freq
    # =========================================================================
    t = torch.arange(seqlen)                    # [0 .. L-1]
    freqs = torch.outer(t, freqs)               # shape: (L, dim/2)

    # =========================================================================
    # Step 4 — Convert to complex exponential representation:
    #         cis(θ) = cos(θ) + i⋅sin(θ)
    #
    # This enables Q and K rotation with:
    #       q_rot = q * cis
    #       k_rot = k * cis
    #
    # where * is complex multiply (implemented via real operations).
    # =========================================================================
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis



def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings (RoPE) to the input tensor using complex-number
    rotation. RoPE injects position-dependent rotation into the hidden representations,
    enabling attention to encode relative positions.

    This implementation:
        • Converts the last dimension of `x` into complex pairs.  
        • Multiplies each complex pair by the corresponding complex exponential from
            `freqs_cis`, performing the rotary transformation.  
        • Converts the rotated complex pairs back to real-valued tensor format.  

    Args:
        x (torch.Tensor):
            The input tensor to which rotary embeddings will be applied.
            Expected shape:
                (batch_size, seq_len, num_heads, head_dim)
            where `head_dim` must be even because RoPE operates on (real, imaginary) pairs.
        
        freqs_cis (torch.Tensor):
            Precomputed complex exponential values encoding positional frequencies.
            Expected shape:
                (seq_len, head_dim // 2)
            Will be automatically broadcast to match the batch and head dimensions.

    Returns:
        torch.Tensor:
            The tensor after applying rotary positional embeddings.
            Shape is identical to the input `x`.

    Notes:
        • The function internally casts `x` to float to perform complex math, then
          restores it to its original dtype.
        • RoPE relies on interpreting each consecutive pair of features as a complex
          number (real, imaginary). Complex multiplication with `freqs_cis` performs
          the rotational transformation.
    """

    # Save original dtype (e.g., bf16, fp16)
    dtype = x.dtype

    # ---------------------------------------------------------------
    # Convert last dimension (head_dim) into complex numbers:
    #   (real_part, imag_part) → complex tensor
    # Shape: (B, T, H, head_dim/2)
    # ---------------------------------------------------------------
    x = torch.view_as_complex(
        x.float().view(*x.shape[:-1], -1, 2)
    )

    # ---------------------------------------------------------------
    # Prepare freqs_cis for broadcasting:
    # Input freqs_cis: (T, head_dim/2)
    # Reshaped to: (1, T, 1, head_dim/2)
    # Broadcasted over: batch_size × num_heads
    # ---------------------------------------------------------------
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))

    # ---------------------------------------------------------------
    # Apply rotation:
    #   complex_out = x * freqs_cis
    # Then convert back to real tensor:
    #   complex → (real, imag) → flatten → original head_dim
    # ---------------------------------------------------------------
    y = torch.view_as_real(x * freqs_cis).flatten(3)

    # Restore original dtype and return
    return y.to(dtype)



class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA)

    MLA is a **factorized-attention mechanism** designed to reduce compute and memory
    cost while preserving the expressive power of multi-head attention.

    Unlike standard attention, MLA:
        • Splits Q/K into "NOPE" (non-positional) and "ROPE" (rotary) subspaces.
        • Uses **low-rank LoRA-style projections** for queries and key/value bases.
        • Stores a compressed representation (kv_lora_rank) for keys/values and only
          expands to full Q/K/V when needed.
        • Supports tensor parallelism via ColumnParallelLinear & RowParallelLinear.
        • Uses two possible attention paths:
            - "naive": standard full-feature attention
            - fused MLA: latent space attention using compressed caches

    Attributes:
        dim: Input hidden dimension.
        n_heads: Total number of attention heads.
        n_local_heads: Heads handled by this device under tensor parallelism.
        q_lora_rank: Low-rank dimension for query factorization.
        kv_lora_rank: Low-rank dimension for key/value compression.
        qk_nope_head_dim: Non-positional Q/K head dimension.
        qk_rope_head_dim: Rotational Q/K head dimension (RoPE).
        v_head_dim: Value head dimension.
        qk_head_dim: Total Q/K dimension = NOPE + ROPE.
        softmax_scale: Scaling factor for Q·Kᵀ / sqrt(d) stabilization.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # ------------------------------
        # Core architectural parameters
        # ------------------------------
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size

        # Low-rank parameters (LoRA-style)
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank

        # Q/K feature splitting (NOPE vs ROPE)
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Value head dimension
        self.v_head_dim = args.v_head_dim

        # -----------------------------------------------------
        # Query projection:
        #   Option A: Direct full Q = xW
        #   Option B: Low-rank Q = Wq_a(x) → RMSNorm → Wq_b
        # -----------------------------------------------------
        if self.q_lora_rank == 0:
            # Direct projection, column-parallel: output spreads across devices
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # Low-rank factorization
            self.wq_a = Linear(self.dim, self.q_lora_rank)     # Down-project
            self.q_norm = RMSNorm(self.q_lora_rank)            # Normalization in latent space
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank, 
                self.n_heads * self.qk_head_dim                # Up-project to full Q dimension
            )

        # -----------------------------------------------------
        # Key/Value low-rank base projection:
        #   wkv_a → produces:
        #       kv_lora_rank (latent K/V bases)
        #       qk_rope_head_dim (ROPE positional K component)
        # -----------------------------------------------------
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)

        # Expand latent K/V into:
        #   NOPE K + V   (per-head)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )

        # Output projection: merges all heads
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # Softmax scaling
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Rescaling for extended RoPE (long sequence correction)
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= mscale * mscale

        # -----------------------------------------------------
        # Cache buffers for incremental generation
        # -----------------------------------------------------
        if attn_impl == "naive":
            # Full expanded cache per head
            self.register_buffer(
                "k_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim),
                persistent=False
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim),
                persistent=False
            )
        else:
            # Compressed latent caches (much smaller)
            self.register_buffer(
                "kv_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
                persistent=False
            )
            self.register_buffer(
                "pe_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
                persistent=False
            )

    # ====================
    # Forward Pass
    # ====================
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Args:
            x (Tensor): Input of shape (B, S, dim).
            start_pos (int): Starting position for writing into cache.
            freqs_cis (Tensor): Precomputed rotary position frequencies.
            mask (Tensor or None): Causal / padding mask.

        Returns:
            Tensor of shape (B, S, dim)
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # ============================================================================
        # 1. Compute Q (with optional low-rank factorization)
        # ============================================================================
        if self.q_lora_rank == 0:
            q = self.wq(x)    # (B, S, n_heads * qk_dim)
        else:
            q_low = self.wq_a(x)           # Down-project
            q_low = self.q_norm(q_low)
            q = self.wq_b(q_low)           # Up-project

        # Reshape into multi-head format
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)

        # Split into NOPE positional & ROPE positional components
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE rotation to q_pe
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # ============================================================================
        # 2. Compute K/V low-rank bases from wkv_a
        # ============================================================================
        kv = self.wkv_a(x)

        # Split into:
        #   kv  → latent K/V base     (kv_lora_rank)
        #   k_pe → positional K       (qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE to positional keys
        # (unsqueeze head dim temporarily for correct broadcasting)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # ============================================================================
        # 3. Attention path A: Naive full-dimensional attention
        # ============================================================================
        if attn_impl == "naive":

            # Combine Q = [q_nope | q_pe]
            q_full = torch.cat([q_nope, q_pe], dim=-1)

            # Expand latent K/V using wkv_b
            kv_expanded = self.wkv_b(self.kv_norm(kv))        # (B, S, n_heads*(nope_dim + vdim))
            kv_expanded = kv_expanded.view(
                bsz, seqlen, self.n_local_heads,
                self.qk_nope_head_dim + self.v_head_dim
            )

            # Split expanded key/value
            k_nope, v = torch.split(
                kv_expanded,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1
            )

            # Combine K = [k_nope | k_pe]
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

            # Save into cache for autoregressive decoding
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v

            # Compute attention scores = Q·Kᵀ
            scores = torch.einsum("bshd,bthd->bsht", q_full, self.k_cache[:bsz, :end_pos])
            scores = scores * self.softmax_scale

        # ============================================================================
        # 4. Attention path B: Fused MLA (latent space attention)
        # ============================================================================
        else:
            # -----------------------------------------------
            # Load dequantized weight for wkv_b if needed
            # -----------------------------------------------
            if self.wkv_b.scale is None:
                wkv_b = self.wkv_b.weight
            else:
                wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)

            # Reshape to per-head blocks
            #   (n_local_heads, out_features_per_head, kv_lora_rank)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

            # -----------------------------------------------
            # Project q_nope via latent key space:
            #    q_nope       : (B, S, H, nope_dim)
            #    wkv_b[:, :nope_dim]  maps latent K to NOPE K space
            # -----------------------------------------------
            q_nope = torch.einsum(
                "bshd,hdc->bshc",
                q_nope,
                wkv_b[:, :self.qk_nope_head_dim]
            )

            # Store latent kv and positional K_ROPE in caches
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

            # -----------------------------------------------
            # Compute attention in latent space:
            #   scores = q_nope · kv_cacheᵀ + q_pe · pe_cacheᵀ
            # -----------------------------------------------
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

        # ============================================================================
        # 5. Apply mask (causal / padding)
        # ============================================================================
        if mask is not None:
            scores = scores + mask.unsqueeze(1)

        # Softmax over "t" dimension (sequence positions)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # ============================================================================
        # 6. Compute output values (V)
        # ============================================================================
        if attn_impl == "naive":
            # Standard: Weighted sum over full V-cache
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # Latent V = scores * kv_cache
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])

            # Expand latent V → full V via wkv_b
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # Flatten heads and project out
        x = self.wo(x.flatten(2))

        return x



class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in DeepSeek-V3 as the feed-forward
    part of the transformer layer.

    This MLP implementation is *not a standard FFN*.  
    It uses:
        - ColumnParallelLinear / RowParallelLinear (tensor parallelism support)
        - A gated activation mechanism:
              silu(W1(x)) * W3(x)
        which is similar to:
            Gated Linear Units (GLU) / SwiGLU
        - High-throughput distributed linear layers for scalability.

    The structure matches the FFN used in modern high-performance LLMs
    (LLaMA, Mixtral, DeepSeek-V3).
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input hidden dimension of the transformer. Also the output dimension (residual path).
            inter_dim (int): Expanded intermediate dimension. Usually 3–4× dim in large LLMs.

        Architecture:
                      ┌─────────────┐
                      │     W1      │  --> "gate branch"
                      └─────────────┘
                            │
                           SiLU
                            │
         x ──► W1 ──► SiLU ──┐
                             │  elementwise multiply
                             ├───► W2 ──► output
         x ──► W3 ───────────┘

        Parallelism:
            * W1, W3 are ColumnParallelLinear
                → split output features among GPUs.

            * W2 is RowParallelLinear
                → split input features among GPUs.

        This gives:
            - Faster training on multi-GPU systems
            - Reduced memory per device
            - Automatic all-reduce synchronization
        """
        super().__init__()

        # First projection (expands dimensionality)
        # ColumnParallelLinear splits OUT features across GPUs.
        self.w1 = ColumnParallelLinear(dim, inter_dim)

        # Second projection (projects back down to hidden dimension)
        # RowParallelLinear splits IN features across GPUs.
        self.w2 = RowParallelLinear(inter_dim, dim)

        # Third projection (second branch of the gated feed-forward)
        # Also ColumnParallelLinear since output is "inter_dim".
        self.w3 = ColumnParallelLinear(dim, inter_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DeepSeek MLP.

        Args:
            x (torch.Tensor): Tensor of shape (batch, seq_len, dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim)

        Computation:

            Let:
                a = W1(x)
                b = W3(x)

            Then the gated activation is:
                h = silu(a) * b

            Then apply the final projection:
                y = W2(h)

            This design is similar to:
                SwiGLU = silu(W1(x)) * W3(x)

            Benefits:
                • Provides better gradient flow than ReLU
                • Allows multiplicative interactions
                • Increases expressiveness without additional depth
                • Reduces training instability

        Distributed behavior:
            • W1 and W3 output slices are placed on each GPU
            • elementwise multiplication happens locally
            • W2 merges (all-reduce) contributions across GPUs
        """
        # Compute W1(x) → gate branch, followed by SiLU activation
        gate = F.silu(self.w1(x))

        # Compute W3(x) → value branch
        value = self.w3(x)

        # Elementwise gated activation (SwiGLU-style)
        hidden = gate * value

        # Final projection W2(hidden)
        # RowParallelLinear performs necessary all-reduce if multiple GPUs
        return self.w2(hidden)



class Gate(nn.Module):
    """
    Gating mechanism used in Mixture-of-Experts (MoE) models.

    The gate selects which experts should process each input token, and with
    what weight. This module implements:
        • Dense score computation (W·x)
        • Optional bias addition
        • Optional grouping of experts (for hierarchical MoE)
        • Top-K routing
        • Softmax/Sigmoid-based gate scoring
        • Optional normalization of routing weights
        • Scaling (route_scale)

    This is a *routing network*, not a normal neural layer. It decides:
        "Which experts should this token go to?"
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the gating module.

        Args:
            args (ModelArgs):
                A configuration object containing the following fields:

                • dim: Hidden dimension of token embeddings
                • n_routed_experts: Total number of experts available
                • n_activated_experts: Number of experts to activate per token (top-K)
                • n_expert_groups: Number of expert groups used for hierarchical routing
                • n_limited_groups: Groups chosen before selecting experts
                • score_func: 'softmax' or 'sigmoid' for score activation
                • route_scale: Scaling factor applied to final routing weights
        """
        super().__init__()

        # Embedding dimension of each token
        self.dim = args.dim

        # Number of experts selected per token (top-K routing)
        self.topk = args.n_activated_experts

        # Grouping parameters (used in multi-group MoE variants)
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups

        # How to convert raw scores → routing weights
        self.score_func = args.score_func

        # Optional scaling factor after normalization
        self.route_scale = args.route_scale

        # ----------------------------------------------------------------------
        # Learnable parameters
        # ----------------------------------------------------------------------
        # Shape: (n_routed_experts, dim)
        # Each expert has its own score vector.
        # Score = W_e · x is the "expert relevance" for token x.
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))

        # Llama uses bias only when dim == 7168 (architecture-specific detail)
        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
            if self.dim == 7168 else None
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs routing for the given tokens.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, dim) or (T, dim), where each row is a token.

        Returns:
            Tuple[Tensor, Tensor]:
                weights:  (B, topk) final normalized routing weights
                indices:  (B, topk) expert indices selected for each token
        """

        # ------------------------------------------------------------------
        # 1. Compute Raw Scores: W · x
        # ------------------------------------------------------------------
        # linear(x, W) performs x @ W^T
        # Output shape: (B, n_routed_experts)
        scores = linear(x, self.weight)

        # ------------------------------------------------------------------
        # 2. Convert raw scores → probability-like values
        # ------------------------------------------------------------------
        if self.score_func == "softmax":
            # Softmax produces a probability distribution across experts
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # Sigmoid produces independent activations per expert
            # (Not normalized — will be normalized later)
            scores = scores.sigmoid()

        # Save pre-bias scores so we can later gather weights
        original_scores = scores

        # ------------------------------------------------------------------
        # 3. Add bias if present (LLAMA-style)
        # ------------------------------------------------------------------
        if self.bias is not None:
            scores = scores + self.bias

        # ------------------------------------------------------------------
        # 4. Multi-Group Routing (Hierarchical MoE)
        # ------------------------------------------------------------------
        # Before selecting individual experts, we optionally restrict routing
        # to only a subset of groups.
        # Example:
        #   Experts = 64, Groups = 8 → 8 groups of 8 experts
        #
        #   Step 1: Select the top `topk_groups`
        #   Step 2: Select experts only *within* those groups
        # ------------------------------------------------------------------
        if self.n_groups > 1:

            # Reshape to: (B, n_groups, experts_per_group)
            scores = scores.view(x.size(0), self.n_groups, -1)

            if self.bias is None:
                # If no bias, use max score in each group as group-level score
                # Shape: (B, n_groups)
                group_scores = scores.amax(dim=-1)
            else:
                # If bias exists, heuristically approximate group strength using
                # sum of top-2 expert scores in each group.
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

            # Select the top-k groups
            # indices shape: (B, topk_groups)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]

            # Mask groups NOT selected
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool)
            mask = mask.scatter_(1, indices, False)

            # Fill out-of-group scores with −∞ so they can never be chosen
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf"))

            # Flatten back to (B, n_routed_experts)
            scores = scores.flatten(1)

        # ------------------------------------------------------------------
        # 5. Select Top-K Experts After Group Filtering
        # ------------------------------------------------------------------
        # indices: (B, topk)
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # Gather the *original* (softmax/sigmoid) values for weights
        weights = original_scores.gather(1, indices)

        # ------------------------------------------------------------------
        # 6. Normalize weights if using sigmoid activation
        # ------------------------------------------------------------------
        if self.score_func == "sigmoid":
            # Convert independent sigmoid activations → normalized mixture
            weights /= weights.sum(dim=-1, keepdim=True)

        # ------------------------------------------------------------------
        # 7. Apply global scaling factor
        # ------------------------------------------------------------------
        weights *= self.route_scale

        # Return both weights and expert indices
        return weights.type_as(x), indices

