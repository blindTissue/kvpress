# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
import random
from transformers import AutoTokenizer
import json
from pathlib import Path

COLORS = [
    "crimson", "navy", "teal", "olive", "maroon",
    "indigo", "gold", "coral", "turquoise", "violet",
    "salmon", "orchid", "tan", "slategray", "plum",
    "khaki", "chocolate", "peru", "lavender", "tomato",
    "seagreen", "darkcyan", "steelblue", "firebrick",
    "mediumvioletred", "rosybrown", "cadetblue", "chartreuse",
    "dimgray", "cornflowerblue", "forestgreen", "sienna",
    "deeppink", "midnightblue", "royalblue", "hotpink",
    "burlywood", "palevioletred", "dodgerblue", "ivory"
]

MENS_APPAREL = [
    "t-shirt", "polo", "dress shirt", "henley", "tank top",
    "sweater", "hoodie", "cardigan", "blazer", "suit jacket",
    "bomber jacket", "denim jacket", "leather jacket", "overcoat",
    "parka", "windbreaker", "peacoat", "raincoat", "waistcoat",
    "jeans", "chinos", "cargo pants", "corduroy pants",
    "shorts", "joggers", "tracksuit",
    "tie", "bow tie", "belt", "scarf", "beanie",
    "cap", "gloves", "socks", "boots",
    "oxfords", "loafers", "sandals", "slippers", "watch",
    "undershirt", "boxers", "swim trunks"
]

ADJECTIVES = [
    "comfortable", "elegant", "lightweight", "durable", "fashionable",
    "versatile", "warm", "breathable", "stretchy", "waterproof",
    "formal", "casual", "cozy", "fitted", "soft",
    "sturdy", "practical", "luxurious", "vintage", "classic",
]

MODEL_NAMES = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it"
]

DEFAULT_NUM_SAMPLES = 1000
DEFAULT_NUM_DISTRACTORS = 19
DEFAULT_SAVE_LOCATION = Path("custom_dataset") / "color_apparel_second_sentence_dataset.json"


@lru_cache(maxsize=None)
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def has_overlap(model_name: str, item_list: list[str], verbose: bool = True) -> bool:
    """
    Check if any items in the list overlap in tokenization.
    """
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


def create_color_apparel_pairs(
    colors: list[str],
    apparels: list[str],
    pair_count: int
) -> list[dict[str, str]]:
    selected_colors = random.sample(colors, pair_count)
    selected_apparels = random.sample(apparels, pair_count)

    pairs = [
        {"color": color, "apparel": apparel}
        for color, apparel in zip(selected_colors, selected_apparels)
    ]

    return pairs


def create_data(
    all_pairs: list[dict[str, str]],
    adj_first: bool = True,
    location_first: bool = True,
    second_sentence: str | None = None,
) -> str:
    """
    Build the context string from a list of color-apparel pairs.

    :param all_pairs: List of dicts with 'color' and 'apparel' keys.
    :param adj_first: If True, "{color} {apparel}"; else "{apparel} of color {color}".
    :param location_first: If True, "In the closet, there are ..."; else "There are ... in the closet."
    :param second_sentence: Optional sentence appended after the pair list
                            (e.g. " The vest also is comfortable.").
    """
    if location_first:
        return_str = "In the closet, there are "
    else:
        return_str = "There are "

    for i in range(len(all_pairs)):
        pair = all_pairs[i]
        if adj_first:
            return_str += f"{pair['color']} {pair['apparel']}"
        else:
            return_str += f"{pair['apparel']} of color {pair['color']}"

        if i < len(all_pairs) - 2:
            return_str += ", "
        elif i == len(all_pairs) - 2:
            return_str += ", and "
        elif i == len(all_pairs) - 1 and not location_first:
            return_str += " in the closet."
    if location_first:
        return_str += "."

    if second_sentence is not None:
        return_str += second_sentence

    return return_str


def create_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_distractors: int = DEFAULT_NUM_DISTRACTORS,
) -> list[dict[str, any]]:
    dataset = []
    for _ in range(num_samples):
        pairs = create_color_apparel_pairs(COLORS, MENS_APPAREL, num_distractors + 1)
        target_index = random.randint(0, num_distractors - 1)
        target_pair = pairs[target_index]

        adjective = random.choice(ADJECTIVES)
        second_sentence = f" The {target_pair['apparel']} also is {adjective}."

        target_pair_with_adj = {
            "color": target_pair["color"],
            "apparel": target_pair["apparel"],
            "adjective": adjective,
        }

        data_adj_first_location_first = create_data(
            pairs, adj_first=True, location_first=True, second_sentence=second_sentence
        )
        data_adj_first_location_last = create_data(
            pairs, adj_first=True, location_first=False, second_sentence=second_sentence
        )
        data_apparel_first_location_first = create_data(
            pairs, adj_first=False, location_first=True, second_sentence=second_sentence
        )
        data_apparel_first_location_last = create_data(
            pairs, adj_first=False, location_first=False, second_sentence=second_sentence
        )

        dataset.append({
            "data_type_1_location_first": data_adj_first_location_first,
            "data_type_1_location_last": data_adj_first_location_last,
            "data_type_2_location_first": data_apparel_first_location_first,
            "data_type_2_location_last": data_apparel_first_location_last,
            "target_pair": target_pair_with_adj,
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
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to {filepath} ({len(dataset)} samples)")


if __name__ == "__main__":
    has_overlap_val = False
    print("Checking for token overlaps...")
    for model_name in MODEL_NAMES:
        overlap_colors = has_overlap(model_name, COLORS)
        overlap_items = has_overlap(model_name, MENS_APPAREL)
        overlap_adjs = has_overlap(model_name, ADJECTIVES)
        if overlap_colors or overlap_items or overlap_adjs:
            has_overlap_val = True
        if overlap_colors:
            print(f"Token overlap found in COLORS for model {model_name}")
        else:
            print(f"No token overlap in COLORS for model {model_name}")
        if overlap_items:
            print(f"Token overlap found in MENS_APPAREL for model {model_name}")
        else:
            print(f"No token overlap in MENS_APPAREL for model {model_name}")
        if overlap_adjs:
            print(f"Token overlap found in ADJECTIVES for model {model_name}")
        else:
            print(f"No token overlap in ADJECTIVES for model {model_name}")

    if not has_overlap_val:
        print("No token overlaps found across all models. Creating dataset...")
        create_and_save_dataset()
    else:
        print("Token overlaps found. Creating dataset anyway (adjective overlap is acceptable)...")
        create_and_save_dataset()
