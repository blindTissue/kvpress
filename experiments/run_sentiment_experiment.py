# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sentiment Locality Experiment for KV Cache.

Tests whether LLMs retain sentiment associations (positive/negative) after
targeted eviction of specific tokens from the KV cache.

5 Conditions:
    1. No eviction (baseline)
    2. Target pair removed from context before prefill (no-info baseline)
    3. Both adjective+noun tokens evicted after prefill
    4. Adjective tokens evicted after prefill (the sentiment carrier)
    5. Noun tokens evicted after prefill (the entity identifier)

1 Question Type:
    "What is the sentiment of {noun}? Answer only in positive or negative."
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import DynamicCache

from experiments.cache_manipulator import clone_cache, evict_from_cache
from experiments.sentiment_context_utils import create_context_without_target
from experiments.sentiment_token_locator import get_context_key, get_eviction_targets
from experiments.run_experiment import (
    load_model_and_tokenizer,
    load_dataset,
    prefill,
    generate_answer,
    jointly_tokenize_and_split,
    prepare_chat_context,
    remove_answer_from_cache,
)

logger = logging.getLogger(__name__)

CONDITION_NAMES = {
    1: "no_eviction",
    2: "removed_from_context",
    3: "evict_both",
    4: "evict_adjective",
    5: "evict_noun",
}


@dataclass
class SentimentExperimentConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_path: str = "custom_dataset/sentiment_dataset.json"
    output_path: str = "experiments/sentiment_results.json"
    data_type: int = 1
    location: str = "first"
    conditions: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    num_samples: int = -1
    max_new_tokens: int = 30
    rerotate: bool = False
    enable_thinking: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"


def build_question(target_pair: dict) -> str:
    return f"What is the sentiment of {target_pair['noun']}? Answer only in positive or negative."


def evaluate_answer(answer: str, target_pair: dict) -> bool:
    """
    Check if the generated answer contains the correct sentiment.

    Returns True only when the correct sentiment word appears and the
    opposite does not, to avoid false positives from hedged answers.
    """
    answer_lower = answer.lower()
    has_positive = "positive" in answer_lower
    has_negative = "negative" in answer_lower
    target = target_pair["sentiment"]

    if target == "positive":
        return has_positive and not has_negative
    else:
        return has_negative and not has_positive


@torch.no_grad()
def run_single_sample(
    sample: dict,
    model,
    tokenizer,
    config: SentimentExperimentConfig,
) -> list[dict]:
    context_key = get_context_key(config.data_type, config.location)
    raw_context = sample[context_key]
    target_pair = sample["target_pair"]
    results = []

    context_text, question_suffix = prepare_chat_context(
        raw_context, tokenizer, enable_thinking=config.enable_thinking
    )

    ref_question = "\n" + build_question(target_pair) + question_suffix
    context_ids, _ = jointly_tokenize_and_split(context_text, ref_question, tokenizer)
    original_context_length = context_ids.shape[1]

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

        raw_question = build_question(target_pair)
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

        correct = evaluate_answer(answer, target_pair)

        results.append({
            "condition": condition,
            "condition_name": CONDITION_NAMES[condition],
            "question": raw_question,
            "answer": answer,
            "correct": correct,
            "target_pair": target_pair,
            "target_index": sample["target_index"],
        })

    return results


def run_experiment(config: SentimentExperimentConfig) -> dict:
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
        all_results.extend(sample_results)

    summary = _compute_summary(all_results)
    _print_summary(summary)

    output = {
        "config": {
            "model_name": config.model_name,
            "data_type": config.data_type,
            "location": config.location,
            "conditions": config.conditions,
            "num_samples": len(dataset),
            "max_new_tokens": config.max_new_tokens,
            "rerotate": config.rerotate,
            "enable_thinking": config.enable_thinking,
            "experiment_type": "sentiment",
        },
        "results": all_results,
        "summary": summary,
    }

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return output


def _compute_summary(results: list[dict]) -> dict:
    from collections import defaultdict

    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        key = r["condition_name"]
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
    print("SENTIMENT EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<35} {'Acc':>8} {'N':>6}")
    print("-" * 60)
    for key, vals in summary.items():
        print(f"{key:<35} {vals['accuracy']:>7.1%} {vals['total']:>6}")
    print("=" * 60)


def parse_args() -> SentimentExperimentConfig:
    parser = argparse.ArgumentParser(description="KV Cache Sentiment Locality Experiment")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="custom_dataset/sentiment_dataset.json")
    parser.add_argument("--output-path", type=str, default="experiments/sentiment_results.json")
    parser.add_argument("--data-type", type=int, default=1, choices=[1, 2])
    parser.add_argument("--location", type=str, default="first", choices=["first", "last"])
    parser.add_argument("--conditions", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-samples", type=int, default=-1, help="-1 for all samples")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--rerotate", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode (e.g. Qwen3). Off by default for direct answers.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    return SentimentExperimentConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        data_type=args.data_type,
        location=args.location,
        conditions=args.conditions,
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
