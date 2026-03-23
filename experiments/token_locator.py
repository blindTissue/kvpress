# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedTokenizerBase


def _get_offset_mapping(
    context: str,
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[int, int]]:
    encoding = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
    return encoding["offset_mapping"]


def _tokens_in_char_range(
    offset_mapping: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> list[int]:
    """Return token indices whose character spans overlap [char_start, char_end)."""
    indices = []
    for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue
        if tok_start < char_end and tok_end > char_start:
            indices.append(token_idx)
    return indices


def find_token_positions(
    context: str,
    substring: str,
    tokenizer: PreTrainedTokenizerBase,
    occurrence: int = 0,
) -> list[int]:
    """
    Find the token indices in a tokenized context that correspond to a given substring.

    Uses the tokenizer's offset mapping to map character spans to token positions.

    Parameters
    ----------
    context : str
        The full context string.
    substring : str
        The substring to locate within the context.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used for tokenization.
    occurrence : int
        Which occurrence of the substring to find (0-indexed). Default is 0 (first).

    Returns
    -------
    list[int]
        Sorted list of token indices whose character spans overlap with the substring.

    Raises
    ------
    ValueError
        If the substring is not found in the context.
    """
    char_start = _find_nth_occurrence(context, substring, occurrence)
    char_end = char_start + len(substring)
    offset_mapping = _get_offset_mapping(context, tokenizer)
    return _tokens_in_char_range(offset_mapping, char_start, char_end)


def _find_nth_occurrence(haystack: str, needle: str, n: int) -> int:
    search_from = 0
    for _ in range(n + 1):
        pos = haystack.find(needle, search_from)
        if pos == -1:
            raise ValueError(
                f"Substring '{needle}' (occurrence {n}) not found in context"
            )
        search_from = pos + 1
    return pos


def _find_pair_char_start(context: str, color: str, apparel: str, data_type: int) -> int:
    """
    Find the character start of the unique pair phrase in the context.

    The full pair phrase is always unique within a sample because both the
    color and the apparel appear exactly once per sample.
    """
    if data_type == 1:
        pair_phrase = f"{color} {apparel}"
    else:
        pair_phrase = f"{apparel} of color {color}"
    pos = context.find(pair_phrase)
    if pos == -1:
        raise ValueError(f"Pair phrase '{pair_phrase}' not found in context")
    return pos


def get_eviction_targets(
    sample: dict,
    data_type: int,
    condition: int,
    tokenizer: PreTrainedTokenizerBase,
    context_text: str,
) -> list[int]:
    """
    Compute token indices to evict for a given dataset sample, data type, and condition.

    For conditions 4 and 5 (single-component eviction), the full pair phrase is
    located first to avoid ambiguous substring matches (e.g. color "tan" matching
    inside apparel "tank top"). Then the character range is narrowed to just the
    target component.

    Parameters
    ----------
    sample : dict
        A single dataset sample containing 'target_pair'.
    data_type : int
        1 for adj-first ("{color} {apparel}"), 2 for apparel-first ("{apparel} of color {color}").
    condition : int
        3 = evict both color and apparel phrase,
        4 = evict color (and connector for type 2),
        5 = evict apparel.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used for the model.
    context_text : str
        The full context string as tokenized (e.g. chat-template-wrapped).
        Token positions are computed against this string.

    Returns
    -------
    list[int]
        Sorted list of token indices to evict from the KV cache.
    """
    if condition not in (3, 4, 5):
        raise ValueError(f"Condition must be 3, 4, or 5 for eviction, got {condition}")

    color = sample["target_pair"]["color"]
    apparel = sample["target_pair"]["apparel"]
    offset_mapping = _get_offset_mapping(context_text, tokenizer)

    if condition == 3:
        if data_type == 1:
            substring = f"{color} {apparel},"
        else:
            substring = f"{apparel} of color {color},"
        char_start = _find_nth_occurrence(context_text, substring, 0)
        return _tokens_in_char_range(offset_mapping, char_start, char_start + len(substring))

    pair_start = _find_pair_char_start(context_text, color, apparel, data_type)

    if data_type == 1:
        # Pair layout: "{color} {apparel}"
        color_start = pair_start
        color_end = pair_start + len(color)
        apparel_start = color_end + 1  # skip the space
        apparel_end = apparel_start + len(apparel)
    else:
        # Pair layout: "{apparel} of color {color}"
        apparel_start = pair_start
        apparel_end = pair_start + len(apparel)
        connector = " of color "
        color_start = apparel_end + len(connector)
        color_end = color_start + len(color)

    if condition == 4:
        if data_type == 1:
            return _tokens_in_char_range(offset_mapping, color_start, color_end)
        else:
            # Evict "of color {color}" (connector included)
            of_color_start = apparel_end + 1  # skip the space after apparel
            return _tokens_in_char_range(offset_mapping, of_color_start, color_end)

    if condition == 5:
        return _tokens_in_char_range(offset_mapping, apparel_start, apparel_end)


def get_context_key(data_type: int, location: str) -> str:
    """Return the dataset sample key for a given data_type and location."""
    return f"data_type_{data_type}_location_{location}"
