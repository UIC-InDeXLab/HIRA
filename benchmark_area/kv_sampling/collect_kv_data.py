"""Utility to sample key/query vectors from a LLaMA-style model for kv sampling tests.

The script loads a causal language model (default: Meta Llama 3) and runs it on a
set of prompts. It captures the key cache for a target layer to build a realistic
sample of key vectors and computes the corresponding query vectors, emitting a
smaller randomly-subsampled set for downstream benchmarks.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import Cache
except Exception:  # pragma: no cover - transformers may be missing during import time
    Cache = None  # type: ignore[assignment]

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "collect_kv_data currently relies on the LLaMA attention implementation. "
        "Please ensure `transformers` is installed with LLaMA support."
    ) from exc


DEFAULT_PROMPTS: Sequence[str] = (
    "Summarize the plot of a mystery novel that happens in Marrakech.",
    "Explain the math concept of Fourier transforms to a senior high school student.",
    "Provide a step-by-step recipe for making Neapolitan pizza at home.",
    "List safety protocols for launching a small satellite.",
    "Write a short dialogue between a spacecraft engineer and a curious child.",
    "Describe the impact of transformers on natural language processing research.",
)


@dataclass
class SamplingConfig:
    model_name: str
    revision: str | None
    layer_index: int
    max_sequence_length: int
    key_sample_size: int
    query_sample_size: int
    prompts: Sequence[str]
    prompt_source: str
    device: str
    dtype: torch.dtype
    output_dir: Path
    prefix: str
    hf_token: str | None
    seed: int


def parse_args() -> SamplingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model id of the LLaMA-style checkpoint.",
    )
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Zero-based layer index to sample keys from (negative indexes allowed).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum number of tokens to keep from every prompt.",
    )
    parser.add_argument(
        "--key-samples",
        type=int,
        default=10000,
        help="Target number of key vectors to keep (samples uniformly if needed).",
    )
    parser.add_argument(
        "--query-samples",
        type=int,
        default=512,
        help="Target number of query vectors to keep (will always be <= key samples).",
    )
    parser.add_argument(
        "--ruler-jsonl",
        type=Path,
        default=None,
        help="Optional path to a RULER dataset JSONL file that provides prompts (overrides --prompt-file).",
    )
    parser.add_argument(
        "--ruler-max-prompts",
        type=int,
        default=64,
        help="Maximum number of prompts to use from the RULER dataset (ignored when --ruler-jsonl is absent).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional path to a text file with one prompt per line.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, defaults to CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
        help="Torch dtype to load the model with.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "kv_data",
        help="Directory for the generated NPZ/JSON files.",
    )
    parser.add_argument(
        "--file-prefix",
        default="kv_data",
        help="Prefix for the saved sample file names.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated checkpoints (env HF_TOKEN is used otherwise).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used when sub-sampling keys/queries.",
    )
    args = parser.parse_args()

    prompts, prompt_source = load_prompts(args.prompt_file, args.ruler_jsonl, args.ruler_max_prompts)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    output_dir = args.output_dir.resolve()
    hf_token = args.hf_token or None

    return SamplingConfig(
        model_name=args.model,
        revision=args.revision,
        layer_index=args.layer,
        max_sequence_length=args.max_seq_length,
        key_sample_size=args.key_samples,
        query_sample_size=args.query_samples,
        prompts=prompts,
        prompt_source=prompt_source,
        device=device,
        dtype=dtype,
        output_dir=output_dir,
        prefix=args.file_prefix,
        hf_token=hf_token,
        seed=args.seed,
    )


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_dtype(name: str) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return lookup[name]


def load_prompts(
    prompt_file: Path | None,
    ruler_jsonl: Path | None,
    ruler_max_prompts: int | None,
) -> tuple[Sequence[str], str]:
    if ruler_jsonl is not None:
        loader = import_ruler_loader()
        resolved_ruler_path = Path(ruler_jsonl).expanduser().resolve()
        if not resolved_ruler_path.is_file():
            raise FileNotFoundError(f"RULER dataset file {resolved_ruler_path} was not found.")
        limit = None if ruler_max_prompts is None or ruler_max_prompts <= 0 else ruler_max_prompts
        prompts = loader(resolved_ruler_path, max_examples=limit)
        if not prompts:
            raise ValueError(f"No prompts found in RULER dataset {resolved_ruler_path}")
        return prompts, f"ruler:{resolved_ruler_path}"

    if prompt_file is None:
        return DEFAULT_PROMPTS, "default"

    lines: List[str] = []
    with Path(prompt_file).open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if stripped:
                lines.append(stripped)
    if not lines:
        raise ValueError(f"No prompts found inside {prompt_file}")
    return lines, str(prompt_file)


def import_ruler_loader():
    try:
        from hira.tests.kv_sampling.ruler_loader import load_ruler_prompts  # type: ignore

        return load_ruler_prompts
    except ModuleNotFoundError:
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from ruler_loader import load_ruler_prompts

        return load_ruler_prompts


def prepare_inputs(
    tokenizer,
    prompt: str,
    device: str,
    max_seq_length: int,
) -> dict:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    if "position_ids" not in encoded:
        attention_mask = encoded["attention_mask"]
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        encoded["position_ids"] = position_ids
    return encoded


def ensure_legacy_cache(past_key_values):
    if past_key_values is None:
        raise RuntimeError("Model did not return past_key_values, cannot capture key cache.")
    if isinstance(past_key_values, tuple):
        return past_key_values
    if Cache is not None and isinstance(past_key_values, Cache):  # pragma: no cover - runtime check
        return past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")


def compute_query_states(
    layer,
    layer_input: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Replicates the LLaMA attention projections to obtain query vectors with RoPE."""
    attn = layer.self_attn
    bsz, seq_len, _ = layer_input.shape
    query_states = attn.q_proj(layer_input)
    key_states = attn.k_proj(layer_input)

    query_states = query_states.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    rotary = getattr(attn, "rotary_emb", None)
    if rotary is not None:
        kv_seq_len = key_states.shape[-2]
        rotary_call = getattr(rotary, "forward", rotary)
        rotary_kwargs = {}
        try:
            signature = inspect.signature(rotary_call)
        except (TypeError, ValueError):  # pragma: no cover - some C++ impls lack signature
            signature = None
        if signature is not None:
            params = signature.parameters
            if "seq_len" in params:
                rotary_kwargs["seq_len"] = kv_seq_len
            if "position_ids" in params:
                rotary_kwargs["position_ids"] = position_ids
        try:
            cos, sin = rotary(key_states, **rotary_kwargs)
        except TypeError:
            # Fall back to positional invocation for older transformers versions.
            if rotary_kwargs.get("position_ids") is not None:
                cos, sin = rotary(key_states, rotary_kwargs["position_ids"])
            else:
                cos, sin = rotary(key_states)
        query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    return query_states


def flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
    last_dim = tensor.shape[-1]
    return tensor.reshape(-1, last_dim).contiguous()


def sample_rows(array: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    if target <= 0 or array.shape[0] <= target:
        return array
    indices = rng.choice(array.shape[0], size=target, replace=False)
    return array[indices]


def collect_samples(cfg: SamplingConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    token_kwargs = {}
    if cfg.hf_token:
        token_kwargs["token"] = cfg.hf_token

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.revision, **token_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.revision,
        torch_dtype=cfg.dtype,
        device_map=None,
        **token_kwargs,
    )
    model.to(cfg.device)
    model.eval()

    key_batches: List[np.ndarray] = []
    query_batches: List[np.ndarray] = []

    selected_layer_index: int | None = None

    with torch.no_grad():
        for prompt in cfg.prompts:
            inputs = prepare_inputs(tokenizer, prompt, cfg.device, cfg.max_sequence_length)
            outputs = model(
                **inputs,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

            cache = ensure_legacy_cache(outputs.past_key_values)
            num_layers = len(cache)
            if selected_layer_index is None:
                selected_layer_index = cfg.layer_index if cfg.layer_index >= 0 else num_layers + cfg.layer_index
                if selected_layer_index < 0 or selected_layer_index >= num_layers:
                    raise IndexError(
                        f"Layer index {cfg.layer_index} is invalid for model with {num_layers} layers."
                    )
            layer_index = selected_layer_index

            layer_keys = cache[layer_index][0].detach().to(torch.float32).to("cpu")
            key_batches.append(flatten_heads(layer_keys).numpy())

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states, can't reconstruct query vectors.")

            layer_input = hidden_states[layer_index]
            queries = compute_query_states(model.model.layers[layer_index], layer_input, inputs["position_ids"])
            query_batches.append(flatten_heads(queries.detach().to(torch.float32).to("cpu")).numpy())

    all_keys = np.concatenate(key_batches, axis=0).astype(np.float32)
    all_queries = np.concatenate(query_batches, axis=0).astype(np.float32)

    sampled_keys = sample_rows(all_keys, cfg.key_sample_size, np_rng)
    target_query_samples = min(cfg.query_sample_size, sampled_keys.shape[0])
    sampled_queries = sample_rows(all_queries, target_query_samples, np_rng)

    if selected_layer_index is None:
        raise RuntimeError("Sampling failed before any prompts were processed.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{cfg.prefix}_{Path(cfg.model_name).name}_layer{selected_layer_index}_{timestamp}"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cfg.output_dir / f"{base}.npz"
    np.savez_compressed(
        npz_path,
        keys=sampled_keys,
        queries=sampled_queries,
    )

    metadata = {
        "model": cfg.model_name,
        "revision": cfg.revision,
        "layer_index": selected_layer_index,
        "num_key_vectors": int(sampled_keys.shape[0]),
        "num_query_vectors": int(sampled_queries.shape[0]),
        "max_sequence_length": cfg.max_sequence_length,
        "prompts": list(cfg.prompts),
        "prompt_source": cfg.prompt_source,
        "dtype": str(cfg.dtype),
        "device": cfg.device,
        "filepath": str(npz_path),
        "generated_at_utc": timestamp,
    }
    metadata_path = cfg.output_dir / f"{base}.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "npz_path": npz_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def main() -> None:
    cfg = parse_args()
    result = collect_samples(cfg)
    print(
        json.dumps(
            {
                "keys": result["metadata"]["num_key_vectors"],
                "queries": result["metadata"]["num_query_vectors"],
                "npz": str(result["npz_path"]),
                "metadata": str(result["metadata_path"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
