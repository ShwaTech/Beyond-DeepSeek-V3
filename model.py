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

