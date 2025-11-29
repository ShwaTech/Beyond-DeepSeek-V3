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
        # 7.6. Early stopping:
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



def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    This function supports:
        - Single-GPU and Multi-GPU (distributed) inference using torch.distributed
        - Interactive REPL generation with a chat-like message history
        - Batch generation from a file
        - Tokenizer integration with chat templates
        - Safe model loading for tensor-parallel checkpoints

    Args:
        ckpt_path (str): Directory containing model weights & tokenizer.
        config (str): Path to model configuration JSON file.
        input_file (str): Optional file containing prompts (batch mode only).
        interactive (bool): Whether to run in REPL/Chat mode.
        max_new_tokens (int): Maximum tokens to generate per prompt.
        temperature (float): Sampling temperature.
    """

    # ------------------------------------------------------------
    # 1. Distributed Inference Setup
    #
    # WORLD_SIZE = number of total processes across all GPUs
    # RANK       = unique ID of this process among all others
    # LOCAL_RANK = index of GPU on this machine
    #
    # These values are set by torchrun or mpirun for multi-GPU execution.
    # ------------------------------------------------------------
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Initialize PyTorch Distributed only if using multiple GPUs.
    # nccl backend = optimized for high-performance GPU-to-GPU communication.
    if world_size > 1:
        dist.init_process_group("nccl")

    # ------------------------------------------------------------
    # 2. Silence printing on non-root ranks.
    #
    # Only rank==0 should print to avoid duplicate outputs.
    #
    # We override print() locally in this function scope.
    # ------------------------------------------------------------
    global print
    if rank != 0:
        print = lambda *_, **__: None  # No-op print

    # ------------------------------------------------------------
    # 3. GPU + PyTorch Runtime Configuration
    #
    # Move this process to its assigned GPU.
    # Set bfloat16 as default numeric dtype (DeepSeek's training default).
    # Use limited CPU threads to avoid oversubscription in inference.
    # ------------------------------------------------------------
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    # ------------------------------------------------------------
    # 4. Load model configuration from JSON and parse into ModelArgs.
    #
    # This provides:
    #   dim, n_layers, n_heads, vocab_size, max_seq_len, etc.
    # ------------------------------------------------------------
    with open(config) as f:
        args = ModelArgs(**json.load(f))

    print(args)

    # ------------------------------------------------------------
    # 5. Construct the Transformer on the correct device (GPU).
    #
    # "with torch.device()" ensures model parameters are created on GPU
    # instead of creating on CPU then transferring to GPU.
    # ------------------------------------------------------------
    with torch.device("cuda"):
        model = Transformer(args)

    # ------------------------------------------------------------
    # 6. Load tokenizer (HuggingFace-compatible).
    #
    # ckpt_path must contain tokenizer files (tokenizer.json, merges.txt, etc.)
    # ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # ------------------------------------------------------------
    # 7. Quick sanity-test generation (optional).
    #
    # Generates 2 tokens for the prompt "DeepSeek" to ensure model & tokenizer work.
    # ------------------------------------------------------------
    tokenizer.decode(
        generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0]
    )

    # ------------------------------------------------------------
    # 8. Load model weights.
    #
    # Model weights are stored as:
    #   model0-mp2.safetensors
    #   model1-mp2.safetensors
    # for tensor-parallel=2, for example.
    #
    # Rank determines which shard to load.
    # ------------------------------------------------------------
    shard_path = os.path.join(
        ckpt_path,
        f"model{rank}-mp{world_size}.safetensors"
    )
    load_model(model, shard_path)

    # ============================================================
    # INTERACTIVE CHAT MODE
    # ============================================================
    if interactive:

        # Maintain conversation memory as a list of:
        #   {"role": "user"/"assistant", "content": "..."}
        messages = []

        while True:

            # ----------------------------------------------------
            # 9.1. Obtain prompt from user.
            #
            # If tensor-parallel > 1, only rank 0 reads input,
            # then it broadcasts the prompt to all ranks.
            # ----------------------------------------------------
            if world_size == 1:
                prompt = input(">>> ")

            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, src=0)

            else:
                objects = [None]
                dist.broadcast_object_list(objects, src=0)
                prompt = objects[0]

            # ----------------------------------------------------
            # 9.2. Handle built-in commands:
            #   /exit → end program
            #   /clear → wipe conversation memory
            # ----------------------------------------------------
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue

            # Add user message to chat history
            messages.append({"role": "user", "content": prompt})

            # ----------------------------------------------------
            # 9.3. Convert chat messages into model tokens.
            #
            # Chat templates define how the sequence is formatted.
            # Example (LLama-style):
            #   <s>[INST] user_message [/INST] assistant_reply
            #
            # add_generation_prompt=True inserts the assistant tag.
            # ----------------------------------------------------
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            # ----------------------------------------------------
            # 9.4. Run autoregressive generation.
            # ----------------------------------------------------
            completion_tokens = generate(
                model,
                [prompt_tokens],            # batch of size 1
                max_new_tokens,
                tokenizer.eos_token_id,
                temperature
            )

            # Decode tokens into text
            completion = tokenizer.decode(
                completion_tokens[0],
                skip_special_tokens=True
            )

            print(completion)

            # Add assistant reply to chat memory
            messages.append({"role": "assistant", "content": completion})

    # ============================================================
    # BATCH GENERATION MODE
    # ============================================================
    else:
        # --------------------------------------------------------
        # 10. Read a list of prompts from file.
        # --------------------------------------------------------
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]

        # Ensure batch does not exceed model capacity
        assert len(prompts) <= args.max_batch_size, \
            f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"

        # --------------------------------------------------------
        # 11. Format each prompt using the chat template.
        # --------------------------------------------------------
        prompt_tokens = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True
            )
            for prompt in prompts
        ]

        # --------------------------------------------------------
        # 12. Generate for whole batch.
        # --------------------------------------------------------
        completion_tokens = generate(
            model,
            prompt_tokens,
            max_new_tokens,
            tokenizer.eos_token_id,
            temperature
        )

        # Decode entire batch
        completions = tokenizer.batch_decode(
            completion_tokens,
            skip_special_tokens=True
        )

        # --------------------------------------------------------
        # 13. Print prompt + generated output for each batch item.
        # --------------------------------------------------------
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    # ------------------------------------------------------------
    # 14. Cleanup distributed resources (if used).
    # ------------------------------------------------------------
    if world_size > 1:
        dist.destroy_process_group()

