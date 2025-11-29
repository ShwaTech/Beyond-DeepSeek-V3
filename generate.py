import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs



def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)



@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Autoregressive text-generation loop for Transformer models.
    Generates new tokens based on the given prompt tokens using the specified model.

    This function takes a batch of tokenized prompts and then generates new tokens
    one-by-one using model.forward() in inference mode. It supports:
        - variable prompt lengths
        - temperature sampling or greedy decoding
        - batch early-stopping at EOS
        - correct handling of prompt tokens vs generated tokens
        - incremental position updates for rotary embeddings

    Args:
        model (Transformer):
            The Transformer model used to generate the next-token logits.

        prompt_tokens (List[List[int]]):
            A list of token sequences (one per batch item). Each prompt may
            differ in length.

        max_new_tokens (int):
            Maximum number of *generated* tokens (not counting the prompt).

        eos_id (int):
            ID of the EOS token. Used for early stopping.

        temperature (float):
            Temperature > 0 → stochastic sampling
            Temperature = 0 → greedy decoding

    Returns:
        List[List[int]]:
            For each batch element, returns only the *generated* portion
            (excluding the prompt) and truncated at EOS if present.
    """

    # ------------------------------------------------------------
    # 1. Compute original prompt lengths for all batch items.
    # ------------------------------------------------------------
    prompt_lens = [len(t) for t in prompt_tokens]

    # Ensure no prompt exceeds the model's maximum context length.
    assert max(prompt_lens) <= model.max_seq_len, \
        f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"

    # ------------------------------------------------------------
    # 2. Determine final sequence length after generation.
    #    total_len = min(model capacity, prompt + new tokens)
    # ------------------------------------------------------------
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))

    # ------------------------------------------------------------
    # 3. Allocate token matrix on GPU.
    #
    # Shape: (batch_size, total_len)
    # Initialize with -1 to mark "unfilled" positions.
    # ------------------------------------------------------------
    tokens = torch.full(
        (len(prompt_tokens), total_len),
        -1,
        dtype=torch.long,
        device="cuda"
    )

    # ------------------------------------------------------------
    # 4. Copy each prompt into the beginning of the token matrix.
    #    Remaining positions stay as -1 until generated.
    # ------------------------------------------------------------
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    # ------------------------------------------------------------
    # 5. prev_pos marks the left boundary of the model input window:
    #    model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    #
    # This is essential for efficient autoregressive generation.
    # ------------------------------------------------------------
    prev_pos = 0

    # Track which sequences have hit EOS (for early batch stopping)
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")

    # ------------------------------------------------------------
    # 6. Create a mask marking prompt vs. generated positions.
    #    prompt_mask[b, t] = True if token is part of prompt
    # ------------------------------------------------------------
    prompt_mask = tokens != -1

    # ------------------------------------------------------------
    # 7. Main autoregressive loop
    #    Starts from min(prompt lengths) because before that,
    #    all sequences are still in the prompt-only zone.
    #
    # For each cur_pos: generate only *that* token.
    # ------------------------------------------------------------
    for cur_pos in range(min(prompt_lens), total_len):

        # --------------------------------------------------------
        # 7.1. Call model.forward() only on the *new window*
        #
        # tokens[:, prev_pos:cur_pos] is the segment being processed
        # prev_pos tells the model where this segment starts
        #
        # Because of rotary embeddings and caching, the model needs
        # the absolute positions of these tokens.
        # --------------------------------------------------------
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

        # --------------------------------------------------------
        # 7.2. Convert logits → next token
        #
        # Temperature > 0 → stochastic sampling
        # Temperature = 0 → greedy (argmax)
        # --------------------------------------------------------
        if temperature > 0:
            next_token = sample(logits, temperature)  # stochastic sampling
        else:
            next_token = logits.argmax(dim=-1)        # greedy decoding

        # --------------------------------------------------------
        # 7.3. Important rule:
        #
        # If cur_pos is *still within* the prompt range for a given
        # sequence, do NOT overwrite prompt tokens.
        #
        # next_token[b] = tokens[b, cur_pos]   (prompt token)
        #
        # This guarantees:
        #   - the prompt is faithfully preserved
        #   - generation happens only *after* the prompt ends
        # --------------------------------------------------------
        next_token = torch.where(
            prompt_mask[:, cur_pos],
            tokens[:, cur_pos],    # keep original prompt token
            next_token             # generate new token
        )

        # Store generated token
        tokens[:, cur_pos] = next_token

        # --------------------------------------------------------
        # 7.4. Mark finished sequences:
        #
        # Condition:
        #   - cur_pos is NOT a prompt position
        #   - token == eos_id
        # --------------------------------------------------------
        finished |= torch.logical_and(
            ~prompt_mask[:, cur_pos],   # only generation zone
            next_token == eos_id
        )

        # --------------------------------------------------------
        # 7.5. Advance prev_pos.
        #     The next forward() call will process only the newly
        #     added token(s).
        # --------------------------------------------------------
        prev_pos = cur_pos

        # --------------------------------------------------------
        # 7. Early stopping:
        #     If ALL sequences have reached EOS, stop generating.
        # --------------------------------------------------------
        if finished.all():
            break

    # ------------------------------------------------------------
    # 8. Extract only the *generated* tokens (discard prompt)
    #    For each batch element:
    #       output = tokens[prompt_len : prompt_len + max_new_tokens]
    #
    #    If EOS appears → truncate at EOS.
    # ------------------------------------------------------------
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # Slice only the generated region
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]

        # Truncate at EOS if present
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]

        completion_tokens.append(toks)

    return completion_tokens

