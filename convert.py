import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file



mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}



def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and shards a HuggingFace-format checkpoint into DeepSeek-style
    parallelized .safetensors files, applying expert-sharding and tensor-parallel
    slicing as required.

    This script is typically used when:
        - you download a checkpoint from HuggingFace
        - but you want to convert it into the custom format required by the
            distributed DeepSeek inference/training codebase.
        - It remaps parameter names, slices tensors along model-parallel dimensions,
            and distributes experts to different tensor-parallel ranks.

    Args:
        hf_ckpt_path (str):
            Path to the directory containing input HuggingFace .safetensors files.

        save_path (str):
            Output directory where the converted & sharded checkpoint files
            will be saved.

        n_experts (int):
            Total number of Mixture-of-Experts experts in the full model.
            Used only if the checkpoint includes MoE layers.

        mp (int):
            Model Parallelism degree (a.k.a. tensor parallelism).
            Example:
                mp = 4 → parameters are sliced into 4 shard files:
                    model0-mp4.safetensors
                    model1-mp4.safetensors
                    model2-mp4.safetensors
                    model3-mp4.safetensors

    Returns:
        None
    """
    # ----------------------------------------------------------------------
    # PERFORMANCE SETTINGS
    # ----------------------------------------------------------------------
    # Increase PyTorch intra-op parallelism for faster CPU checkpoint loading.
    torch.set_num_threads(8)

    # Number of experts *per* tensor-parallel rank.
    # Example:
    #   If total experts = 64 and mp = 4
    #       → each rank gets 16 experts.
    n_local_experts = n_experts // mp

    # Create empty dictionaries where each rank's sliced parameters will be stored.
    # Example: state_dicts[0] → parameters for model0-mp{mp}.safetensors
    state_dicts = [{} for _ in range(mp)]

    # ----------------------------------------------------------------------
    # LOAD EACH SAFETENSORS FILE FROM THE HF CHECKPOINT DIRECTORY
    # ----------------------------------------------------------------------
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        # safe_open allows reading tensors WITHOUT loading everything into memory.
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # Iterate over all tensor parameter names inside the file.
            for name in f.keys():

                # Skip known-broken or unused parameters.
                if "model.layers.61" in name:
                    continue

                # Extract tensor from file (still stays on CPU).
                param: torch.Tensor = f.get_tensor(name)

                # HuggingFace checkpoints prefix all parameters with "model." → remove it.
                if name.startswith("model."):
                    name = name[len("model."):]

                # UNIFY PARAMETER NAMING SCHEME
                # These conversions rename keys to match DeepSeek's internal model format.
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")

                # Identify which parameter group this name belongs to.
                # Example:
                #   "layers.0.attn.q_proj.weight"
                # → key = "q_proj"
                key = name.split(".")[-2]

                # Validate that mapping exists in the global parameter-remapping table.
                assert key in mapping, f"Key {key} not found in mapping"

                # mapping[key] returns:
                #   new_key : the renamed parameter key
                #   dim     : the dimension along which model-parallel slicing is applied
                new_key, dim = mapping[key]

                # Replace the old key with the new unified name.
                name = name.replace(key, new_key)

                # ------------------------------------------------------------------
                # SHARDING LOGIC — DISTRIBUTE TENSORS ACROSS mp RANKS
                # ------------------------------------------------------------------
                for i in range(mp):
                    # Default: no slicing (for parameters that are replicated)
                    new_param = param

                    # -------------------------------
                    # CASE 1 — MoE EXPERT-SPECIFIC PARAMETERS
                    # -------------------------------
                    if "experts" in name and "shared_experts" not in name:
                        # Extract the expert index from the name.
                        # Example:
                        #   "layers.0.ffn.experts.14.w1.weight" → idx = 14
                        idx = int(name.split(".")[-3])

                        # Each rank *only owns* its local subset of experts:
                        # Rank 0 → experts 0 – (n_local_experts-1)
                        # Rank 1 → next chunk, etc.
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            # Not this rank → skip
                            continue

                    # -------------------------------
                    # CASE 2 — TENSORS WITH MODEL-PARALLEL DIMENSION
                    # -------------------------------
                    elif dim is not None:
                        # Validate the tensor dimension is divisible by tensor parallelism.
                        assert param.size(dim) % mp == 0, (
                            f"Dimension {dim} must be divisible by mp={mp}"
                        )

                        # Compute slice size (shard size) along the parallel dimension.
                        shard_size = param.size(dim) // mp

                        # Slice the tensor for this rank.
                        # Example:
                        #   rank i gets param[..., i*shard : (i+1)*shard]
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()

                    # -------------------------------
                    # SAVE THE PARAMETER SHARD FOR THIS RANK
                    # -------------------------------
                    state_dicts[i][name] = new_param

    # ----------------------------------------------------------------------
    # SAVE THE SHARDED CHECKPOINT FILES
    # ----------------------------------------------------------------------
    # Ensure output directory exists.
    os.makedirs(save_path, exist_ok=True)

    # For each tensor parallel rank, save one safetensors file.
    for i in trange(mp):
        save_file(
            state_dicts[i],
            os.path.join(save_path, f"model{i}-mp{mp}.safetensors")
        )

    # ----------------------------------------------------------------------
    # COPY TOKENIZER FILES
    # ----------------------------------------------------------------------
    # All "tokenizer.model" or "tokenizer.json" files should be preserved.
    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
