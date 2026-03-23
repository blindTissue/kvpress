# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Information Locality Experiment for KV Cache.

Tests whether LLMs retain color-apparel associations after targeted eviction
of specific tokens from the KV cache.

5 Conditions:
    1. No eviction (baseline)
    2. Target pair removed from context before prefill (no-info baseline)
    3. Both color+apparel tokens evicted after prefill
    4. Color tokens evicted after prefill
    5. Apparel tokens evicted after prefill

2 Question Types:
    1. "What is the color of {apparel}?"
    2. "What is the apparel of color {color}?"
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from experiments.cache_manipulator import clone_cache, evict_from_cache
from experiments.context_utils import create_context_without_target
from experiments.token_locator import get_context_key, get_eviction_targets

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

CONDITION_NAMES = {
    1: "no_eviction",
    2: "removed_from_context",
    3: "evict_both",
    4: "evict_color",
    5: "evict_apparel",
}

QUESTION_TYPE_NAMES = {
    1: "ask_color",
    2: "ask_apparel",
}


@dataclass
class ExperimentConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_path: str = "custom_dataset/color_apparel_dataset.json"
    output_path: str = "experiments/results.json"
    data_type: int = 1
    location: str = "first"
    conditions: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    question_types: list[int] = field(default_factory=lambda: [1, 2])
    num_samples: int = -1
    max_new_tokens: int = 30
    rerotate: bool = False
    enable_thinking: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"


def load_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def load_model_and_tokenizer(config: ExperimentConfig):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        device_map=config.device,
    )
    model.eval()
    return model, tokenizer


def prepare_chat_context(
    raw_context: str,
    tokenizer: PreTrainedTokenizerBase,
    enable_thinking: bool = False,
) -> tuple[str, str]:
    """
    Wrap a raw context string with the tokenizer's chat template.

    Uses the same separator trick as KVPressTextGenerationPipeline.preprocess:
    the context is placed inside a user message, and the result is split into
    the context prefix (everything before the user content ends) and the
    question suffix (the assistant prompt header that follows).

    Parameters
    ----------
    raw_context : str
        The plain-text context (e.g. "In the closet, there are ...").
    tokenizer : PreTrainedTokenizerBase
    enable_thinking : bool
        Whether to enable thinking mode in the chat template. When False,
        models like Qwen3 include an empty <think></think> block in the
        suffix to suppress chain-of-thought and produce direct answers.

    Returns
    -------
    tuple[str, str]
        (context_text, question_suffix) where context_text is ready to be
        tokenized for prefill and question_suffix should be appended after
        each question string before tokenizing.
    """
    if tokenizer.chat_template is None:
        bos_token = getattr(tokenizer, "bos_token", "")
        return bos_token + raw_context, "\n"

    separator = "#" * (len(raw_context) + 10)

    kwargs = dict(
        add_generation_prompt=True,
        tokenize=False,
    )
    # Only pass enable_thinking if the template supports it (e.g. Qwen3),
    # otherwise some templates raise on unexpected kwargs.
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_context + separator}],
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_context + separator}],
            **kwargs,
        )

    context_text, question_suffix = formatted.split(separator)
    return context_text, question_suffix


@torch.no_grad()
def prefill(model, context_ids: torch.Tensor) -> DynamicCache:
    """Run prefill (backbone only, no LM head) and return the populated cache."""
    cache = DynamicCache()
    model.model(
        input_ids=context_ids.to(model.device),
        past_key_values=cache,
    )
    return cache


def jointly_tokenize_and_split(
    context_text: str,
    question_with_suffix: str,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize context + question as a single string, then split the token IDs
    at the context/question boundary.

    BPE tokenization is not compositional: tokenizing "A" and "B" separately
    can produce different tokens at the boundary than tokenizing "AB" together
    (e.g. ".\\n" may be one token jointly but two tokens separately). This
    function ensures consistent tokenization by always tokenizing the full
    string and splitting afterwards.

    Returns (context_ids, question_ids) as 2-D tensors with batch dim.
    """
    full_text = context_text + question_with_suffix
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    offsets = tokenizer(
        full_text, return_offsets_mapping=True, add_special_tokens=False
    )["offset_mapping"]

    ctx_char_end = len(context_text)
    split_idx = next(i for i, (s, _e) in enumerate(offsets) if s >= ctx_char_end)

    context_ids = torch.tensor([full_ids[:split_idx]])
    question_ids = torch.tensor([full_ids[split_idx:]])
    return context_ids, question_ids


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question_text: str,
    cache: DynamicCache,
    context_length: int,
    max_new_tokens: int = 30,
    question_ids: torch.Tensor = None,
) -> str:
    """
    Generate an answer via greedy decoding, mirroring KVPressTextGenerationPipeline.generate_answer.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    question_text : str
        The question string to encode and feed after the cached context.
        Ignored when ``question_ids`` is provided.
    cache : DynamicCache
        The KV cache (already populated with context).
    context_length : int
        Position offset for the first question token. When rerotate=False this
        is the *original* context length; when rerotate=True it equals
        cache.get_seq_length().
    max_new_tokens : int
    question_ids : torch.Tensor, optional
        Pre-tokenized question token IDs (2-D with batch dim). When provided,
        ``question_text`` is ignored. Use ``jointly_tokenize_and_split`` to
        obtain correctly-tokenized IDs.

    Returns
    -------
    str
        The decoded generated text (question tokens excluded).
    """
    if question_ids is None:
        question_ids = tokenizer.encode(question_text, return_tensors="pt", add_special_tokens=False)
    question_ids = question_ids.to(model.device)

    position_ids = torch.arange(
        context_length, context_length + question_ids.shape[1], device=model.device
    ).unsqueeze(0)

    outputs = model(
        input_ids=question_ids,
        past_key_values=cache,
        position_ids=position_ids,
        num_logits_to_keep=1,
    )

    position_ids = position_ids[:, -1:] + 1
    generated_ids = [outputs.logits[0, -1].argmax()]

    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    for i in range(max_new_tokens - 1):
        outputs = model(
            input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
            past_key_values=cache,
            position_ids=position_ids + i,
        )
        new_id = outputs.logits[0, -1].argmax()
        generated_ids.append(new_id)
        if new_id.item() in eos_ids:
            break

    return tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)


def remove_answer_from_cache(cache: DynamicCache, seq_lengths: list[int]):
    """Strip generated/question tokens from cache, restoring it to context-only state."""
    for layer_idx, length in enumerate(seq_lengths):
        cache.layers[layer_idx].keys = cache.layers[layer_idx].keys[:, :, :length]
        cache.layers[layer_idx].values = cache.layers[layer_idx].values[:, :, :length]


def build_question(question_type: int, target_pair: dict) -> str:
    if question_type == 1:
        return f"What is the color of {target_pair['apparel']}?"
    else:
        return f"What is the apparel of color {target_pair['color']}?"


def evaluate_answer(answer: str, target_pair: dict, question_type: int) -> bool:
    """Check if the generated answer contains the correct target word."""
    answer_lower = answer.lower()
    if question_type == 1:
        return target_pair["color"].lower() in answer_lower
    else:
        return target_pair["apparel"].lower() in answer_lower


@torch.no_grad()
def run_single_sample(
    sample: dict,
    model,
    tokenizer,
    config: ExperimentConfig,
) -> list[dict]:
    """
    Run all requested conditions and question types for a single dataset sample.

    Returns a list of result dicts, one per (condition, question_type) pair.
    """
    context_key = get_context_key(config.data_type, config.location)
    raw_context = sample[context_key]
    target_pair = sample["target_pair"]
    results = []

    # Apply chat template to get properly formatted context and question suffix
    context_text, question_suffix = prepare_chat_context(
        raw_context, tokenizer, enable_thinking=config.enable_thinking
    )

    # Use joint tokenization to avoid BPE boundary mismatches between
    # context and question (e.g. ".\n" merging into one token).
    ref_question = "\n" + build_question(config.question_types[0], target_pair) + question_suffix
    context_ids, _ = jointly_tokenize_and_split(context_text, ref_question, tokenizer)
    original_context_length = context_ids.shape[1]

    # Prefill once with full context (reused for conditions 1, 3, 4, 5)
    needs_full_prefill = any(c in config.conditions for c in [1, 3, 4, 5])
    base_cache = None
    if needs_full_prefill:
        base_cache = prefill(model, context_ids)

    for condition in config.conditions:
        if condition == 1:
            cache = clone_cache(base_cache)
            ctx_len = original_context_length
            cond_context_text = context_text

        elif condition == 2:
            modified_raw = create_context_without_target(
                sample, config.data_type, config.location
            )
            modified_text, _ = prepare_chat_context(
                modified_raw, tokenizer, enable_thinking=config.enable_thinking
            )
            modified_ids, _ = jointly_tokenize_and_split(
                modified_text, ref_question, tokenizer
            )
            cache = prefill(model, modified_ids)
            ctx_len = modified_ids.shape[1]
            cond_context_text = modified_text

        elif condition in (3, 4, 5):
            cache = clone_cache(base_cache)
            eviction_positions = get_eviction_targets(
                sample, config.data_type, condition, tokenizer, context_text
            )
            cache, orig_len = evict_from_cache(
                cache,
                eviction_positions,
                rerotate=config.rerotate,
                model=model if config.rerotate else None,
            )
            if config.rerotate:
                ctx_len = cache.get_seq_length()
            else:
                ctx_len = original_context_length
            cond_context_text = context_text
        else:
            raise ValueError(f"Unknown condition: {condition}")

        for question_type in config.question_types:
            raw_question = build_question(question_type, target_pair)
            question_with_suffix = "\n" + raw_question + question_suffix

            _, q_ids = jointly_tokenize_and_split(
                cond_context_text, question_with_suffix, tokenizer
            )

            cache_seq_lengths = [
                cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))
            ]

            answer = generate_answer(
                model, tokenizer, None, cache, ctx_len, config.max_new_tokens,
                question_ids=q_ids,
            )

            remove_answer_from_cache(cache, cache_seq_lengths)

            correct = evaluate_answer(answer, target_pair, question_type)

            results.append({
                "condition": condition,
                "condition_name": CONDITION_NAMES[condition],
                "question_type": question_type,
                "question_type_name": QUESTION_TYPE_NAMES[question_type],
                "question": raw_question,
                "answer": answer,
                "correct": correct,
                "target_pair": target_pair,
                "target_index": sample["target_index"],
            })

    return results


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run the full information locality experiment.

    Returns
    -------
    dict
        Contains 'config', 'results' (per-sample), and 'summary' (accuracy tables).
    """
    logger.info(f"Loading model: {config.model_name}")
    model, tokenizer = load_model_and_tokenizer(config)

    logger.info(f"Loading dataset: {config.dataset_path}")
    dataset = load_dataset(config.dataset_path)

    if config.num_samples > 0:
        dataset = dataset[: config.num_samples]

    all_results = []
    for sample_idx, sample in enumerate(dataset):
        if sample_idx % 50 == 0:
            logger.info(f"Processing sample {sample_idx}/{len(dataset)}")

        sample_results = run_single_sample(sample, model, tokenizer, config)
        for r in sample_results:
            r["sample_idx"] = sample_idx
        all_results.extend(sample_results)

    summary = _compute_summary(all_results)

    output = {
        "config": {
            "model_name": config.model_name,
            "data_type": config.data_type,
            "location": config.location,
            "conditions": config.conditions,
            "question_types": config.question_types,
            "num_samples": len(dataset),
            "rerotate": config.rerotate,
            "enable_thinking": config.enable_thinking,
            "max_new_tokens": config.max_new_tokens,
        },
        "results": all_results,
        "summary": summary,
    }

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    _print_summary(summary)
    return output


def _compute_summary(results: list[dict]) -> dict:
    """Aggregate per-condition, per-question-type accuracy."""
    from collections import defaultdict

    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        key = f"{r['condition_name']}__{r['question_type_name']}"
        counts[key]["total"] += 1
        if r["correct"]:
            counts[key]["correct"] += 1

    summary = {}
    for key, vals in sorted(counts.items()):
        accuracy = vals["correct"] / vals["total"] if vals["total"] > 0 else 0.0
        summary[key] = {
            "correct": vals["correct"],
            "total": vals["total"],
            "accuracy": round(accuracy, 4),
        }
    return summary


def _print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Condition + Question Type':<45} {'Acc':>8} {'N':>6}")
    print("-" * 60)
    for key, vals in summary.items():
        condition, qtype = key.split("__")
        label = f"{condition} / {qtype}"
        print(f"{label:<45} {vals['accuracy']:>7.1%} {vals['total']:>6}")
    print("=" * 60)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="KV Cache Information Locality Experiment")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="custom_dataset/color_apparel_dataset.json")
    parser.add_argument("--output-path", type=str, default="experiments/results.json")
    parser.add_argument("--data-type", type=int, default=1, choices=[1, 2])
    parser.add_argument("--location", type=str, default="first", choices=["first", "last"])
    parser.add_argument("--conditions", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--question-types", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--num-samples", type=int, default=-1, help="-1 for all samples")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--rerotate", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode (e.g. Qwen3). Off by default for direct answers.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    return ExperimentConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        data_type=args.data_type,
        location=args.location,
        conditions=args.conditions,
        question_types=args.question_types,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        rerotate=args.rerotate,
        enable_thinking=args.enable_thinking,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = parse_args()
    run_experiment(config)
