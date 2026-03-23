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
DEFAULT_SAVE_LOCATION = Path("custom_dataset") / "color_apparel_dataset.json"
@lru_cache(maxsize=None)
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

# Here since the original implementation checked for the first token correctness.
# Also useful for checking token position for eviction.
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
    """
    Create pairs of colors and apparel items.
    
    Returns:
        List of Dicts with 'color' and 'apparel' keys.
    """
    selected_colors = random.sample(colors, pair_count)
    selected_apparels = random.sample(apparels, pair_count)
    
    pairs = [
        {"color": color, "apparel": apparel}
        for color, apparel in zip(selected_colors, selected_apparels)
    ]
    
    return pairs

def create_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_distractors: int = DEFAULT_NUM_DISTRACTORS,
) -> list[dict[str, any]]:
    """
    Create a dataset of color-apparel pairs with distractors.
       
    """
    dataset = []
    for _ in range(num_samples):
        pairs = create_color_apparel_pairs(COLORS, MENS_APPAREL, num_distractors + 1)
        target_index = random.randint(0, num_distractors - 1) # Ensure target is not the last one. Dealing with and token
        target_pair = pairs[target_index]
        data_adj_first_location_first = create_data(pairs, adj_first=True, location_first=True)
        data_adj_first_location_last = create_data(pairs, adj_first=True, location_first=False)
        data_apparel_first_location_first = create_data(pairs, adj_first=False, location_first=True)
        data_apparel_first_location_last = create_data(pairs, adj_first=False, location_first=False)
        
        
        dataset.append({
            "data_type_1_location_first": data_adj_first_location_first,
            "data_type_1_location_last": data_adj_first_location_last,
            "data_type_2_location_first": data_apparel_first_location_first,
            "data_type_2_location_last": data_apparel_first_location_last,
            "target_pair": target_pair,
            "pair_list": pairs,
            "target_index": target_index,
            "total_pair_count": len(pairs)
        })
        
    return dataset

def create_data(
    all_pairs: list[dict[str, str]],
    adj_first: bool = True,
    location_first: bool = True
) -> str:
    """
    Docstring for create_data
    
    :param all_pairs: A list of dictionaries, where each dictionary contains a 'color' and 'apparel' key representing a color-apparel pair.
    :param adj_first: Whether to use format with adjectives first (e.g., "There are {color} {apparel}...") or with apparel first (e.g., "There are {apparel} of color {color}...").
    :type adj_first: bool
    :param location_first: Whether to place the location of objects at the beginning of the sentence (e.g., "In the closet, there are...") or at the end (e.g., "There are... in the closet.").
    :type location_first: bool
    
    :return: A string describing the color-apparel pairs in the closet.
    We might edit this function in future iteration.
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
    return return_str
        
        


# def create_data_type_1(target_pair: dict[str, str], all_pairs: list[dict[str, str]]) -> str:
#     """
#     Create data type 1 with following format:
#     There are {color} {apparel}, {color} {apparel}, ..., and {color} {apparel} in the closet. 
    
#     :param target_pair: Description
#     :type target_pair: dict[str, str]
#     :param all_pairs: Description
#     :type all_pairs: list[dict[str, str]]
#     :return: Description
#     :rtype: str
#     """
#     text = "There are "
#     for i, pair in enumerate(all_pairs):
#         text += f"{pair['color']} {pair['apparel']}"
#         if i < len(all_pairs) - 2:
#             text += ", "
#         elif i == len(all_pairs) - 2:
#             text += ", and "
#         else:
#             text += " in the closet."
#     return text

    
    
# def create_data_type_2(target_pair: dict[str, str], all_pairs: list[dict[str, str]]) -> str:
#     """
#     Create data type 2 with following format:
#     There are {apparel} of color {color}, {apparel} of color {color}, ..., and {apparel} of color {color} in the closet.
#     """
    
#     text = "There are "
#     for i, pair in enumerate(all_pairs):
#         text += f"{pair['apparel']} of color {pair['color']}"
#         if i < len(all_pairs) - 2:
#             text += ", "
#         elif i == len(all_pairs) - 2:
#             text += ", and "
#         else:
#             text += " in the closet."
#     return text

def create_and_save_dataset(
    filepath: str = DEFAULT_SAVE_LOCATION,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_distractors: int = DEFAULT_NUM_DISTRACTORS,
):
    
    dataset = create_dataset(num_samples, num_distractors)
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    
if __name__ == "__main__":
    has_overlap_val = False
    print("Checking for token overlaps...")
    for model_name in MODEL_NAMES:
        overlap_colors = has_overlap(model_name, COLORS)
        overlap_items = has_overlap(model_name, MENS_APPAREL)
        if overlap_colors or overlap_items:
            has_overlap_val = True
        if overlap_colors:
            print(f"Token overlap found in COLORS for model {model_name}")
        else:
            print(f"No token overlap in COLORS for model {model_name}")
        if overlap_items:
            print(f"Token overlap found in MENS_APPAREL for model {model_name}")
        else:
            print(f"No token overlap in MENS_APPAREL for model {model_name}")
    
    
    if not has_overlap_val:
        print("No token overlaps found across all models. Creating dataset...")
        create_and_save_dataset()