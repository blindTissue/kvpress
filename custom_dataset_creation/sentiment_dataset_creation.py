# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
import random
from transformers import AutoTokenizer
import json
from pathlib import Path

POSITIVE_ADJECTIVES = [
    "delightful", "wonderful", "magnificent", "brilliant", "exquisite",
    "superb", "splendid", "marvelous", "elegant", "gorgeous",
    "charming", "lovely", "pristine", "flawless", "stunning",
    "graceful", "radiant", "dazzling", "enchanting", "immaculate",
]

NEGATIVE_ADJECTIVES = [
    "terrible", "dreadful", "hideous", "appalling", "ghastly",
    "horrendous", "wretched", "disgusting", "repulsive", "atrocious",
    "revolting", "abominable", "grotesque", "shabby", "decrepit",
    "miserable", "awful", "dismal", "repugnant", "tattered",
]

ALL_ADJECTIVES = POSITIVE_ADJECTIVES + NEGATIVE_ADJECTIVES

NEUTRAL_NOUNS = [
    "cake", "painting", "vase", "lamp", "rug",
    "chair", "curtain", "blanket", "mirror", "clock",
    "book", "table", "shelf", "pillow", "candle",
    "basket", "bowl", "mug", "plate", "frame",
    "stool", "bench", "quilt", "tapestry", "figurine",
    "statue", "ornament", "chest", "pot", "tray",
    "cup", "bottle", "jug", "pitcher", "bucket",
    "carpet", "sculpture", "photograph", "sketch", "trophy",
]

MODEL_NAMES = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
]

DEFAULT_NUM_SAMPLES = 1000
DEFAULT_NUM_DISTRACTORS = 19
DEFAULT_SAVE_LOCATION = Path("custom_dataset") / "sentiment_dataset.json"


@lru_cache(maxsize=None)
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def has_overlap(model_name: str, item_list: list[str], verbose: bool = True) -> bool:
    tokenizer = get_tokenizer(model_name)
    token_set = set()

    for item in item_list:
        tokens = tokenizer.encode(item, add_special_tokens=False)
        tokens_with_space = tokenizer.encode(" " + item, add_special_tokens=False)

        first_token = tokens[0] if tokens else None
        first_token_with_space = tokens_with_space[0] if tokens_with_space else None

        if first_token in token_set or first_token_with_space in token_set:
            if verbose:
                print(f"Overlap found for item: '{item}'")
            return True

        if first_token is not None:
            token_set.add(first_token)
        if first_token_with_space is not None:
            token_set.add(first_token_with_space)

    return False


def create_sentiment_pairs(
    nouns: list[str],
    positive_adjs: list[str],
    negative_adjs: list[str],
    pair_count: int,
) -> list[dict[str, str]]:
    selected_nouns = random.sample(nouns, pair_count)

    n_positive = pair_count // 2
    n_negative = pair_count - n_positive
    selected_pos = random.sample(positive_adjs, n_positive)
    selected_neg = random.sample(negative_adjs, n_negative)

    adjectives = selected_pos + selected_neg
    sentiments = ["positive"] * n_positive + ["negative"] * n_negative

    combined = list(zip(adjectives, sentiments))
    random.shuffle(combined)
    adjectives, sentiments = zip(*combined)

    return [
        {"noun": noun, "adjective": adj, "sentiment": sent}
        for noun, adj, sent in zip(selected_nouns, adjectives, sentiments)
    ]


def create_data(
    all_pairs: list[dict[str, str]],
    adj_first: bool = True,
    location_first: bool = True,
) -> str:
    """
    Build a context string from noun-adjective pairs.

    Type 1 (adj_first=True):  "{adjective} {noun}"      e.g. "delightful cake"
    Type 2 (adj_first=False): "{noun} that was {adjective}" e.g. "cake that was delightful"
    """
    if location_first:
        return_str = "In the room, there are "
    else:
        return_str = "There are "

    for i, pair in enumerate(all_pairs):
        if adj_first:
            return_str += f"{pair['adjective']} {pair['noun']}"
        else:
            return_str += f"{pair['noun']} that was {pair['adjective']}"

        if i < len(all_pairs) - 2:
            return_str += ", "
        elif i == len(all_pairs) - 2:
            return_str += ", and "
        elif i == len(all_pairs) - 1 and not location_first:
            return_str += " in the room."

    if location_first:
        return_str += "."
    return return_str


def create_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_distractors: int = DEFAULT_NUM_DISTRACTORS,
) -> list[dict]:
    dataset = []
    for _ in range(num_samples):
        pairs = create_sentiment_pairs(
            NEUTRAL_NOUNS, POSITIVE_ADJECTIVES, NEGATIVE_ADJECTIVES, num_distractors + 1
        )
        target_index = random.randint(0, num_distractors - 1)
        target_pair = pairs[target_index]

        data = {}
        for adj_first in [True, False]:
            for location_first in [True, False]:
                dt = 1 if adj_first else 2
                loc = "first" if location_first else "last"
                data[f"data_type_{dt}_location_{loc}"] = create_data(
                    pairs, adj_first=adj_first, location_first=location_first
                )

        dataset.append({
            **data,
            "target_pair": target_pair,
            "pair_list": pairs,
            "target_index": target_index,
            "total_pair_count": len(pairs),
        })

    return dataset


def create_and_save_dataset(
    filepath: str = DEFAULT_SAVE_LOCATION,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_distractors: int = DEFAULT_NUM_DISTRACTORS,
):
    dataset = create_dataset(num_samples, num_distractors)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Saved {len(dataset)} samples to {filepath}")


if __name__ == "__main__":
    has_overlap_val = False
    print("Checking for token overlaps...")
    for model_name in MODEL_NAMES:
        overlap_nouns = has_overlap(model_name, NEUTRAL_NOUNS)
        overlap_adjs = has_overlap(model_name, ALL_ADJECTIVES)
        if overlap_nouns or overlap_adjs:
            has_overlap_val = True
        if overlap_nouns:
            print(f"Token overlap found in NOUNS for model {model_name}")
        else:
            print(f"No token overlap in NOUNS for model {model_name}")
        if overlap_adjs:
            print(f"Token overlap found in ADJECTIVES for model {model_name}")
        else:
            print(f"No token overlap in ADJECTIVES for model {model_name}")

    if not has_overlap_val:
        print("No token overlaps found across all models. Creating dataset...")
        create_and_save_dataset()
    else:
        print("Token overlaps found. Please review word lists before creating dataset.")
        print("Creating dataset anyway (overlaps may not affect sentiment evaluation)...")
        create_and_save_dataset()
