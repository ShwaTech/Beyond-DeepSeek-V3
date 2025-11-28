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
