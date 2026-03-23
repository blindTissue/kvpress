# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedTokenizerBase

from experiments.token_locator import (
    _get_offset_mapping,
    _tokens_in_char_range,
    _find_nth_occurrence,
)


def _find_pair_char_start(context: str, adjective: str, noun: str, data_type: int) -> int:
    """
    Find the character start of the unique pair phrase in the context.

    Type 1: "{adjective} {noun}"
    Type 2: "{noun} that was {adjective}"
    """
    if data_type == 1:
        pair_phrase = f"{adjective} {noun}"
    else:
        pair_phrase = f"{noun} that was {adjective}"
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
    Compute token indices to evict for a sentiment sample.

    Conditions
    ----------
    3 : evict entire "{adjective} {noun}," or "{noun} that was {adjective},"
    4 : evict adjective only (the sentiment-carrying word);
        for type 2, evicts "that was {adjective}" (connector included)
    5 : evict noun only (the entity identifier)
    """
    if condition not in (3, 4, 5):
        raise ValueError(f"Condition must be 3, 4, or 5 for eviction, got {condition}")

    adjective = sample["target_pair"]["adjective"]
    noun = sample["target_pair"]["noun"]
    offset_mapping = _get_offset_mapping(context_text, tokenizer)

    if condition == 3:
        if data_type == 1:
            substring = f"{adjective} {noun},"
        else:
            substring = f"{noun} that was {adjective},"
        char_start = _find_nth_occurrence(context_text, substring, 0)
        return _tokens_in_char_range(offset_mapping, char_start, char_start + len(substring))

    pair_start = _find_pair_char_start(context_text, adjective, noun, data_type)

    if data_type == 1:
        adj_start = pair_start
        adj_end = pair_start + len(adjective)
        noun_start = adj_end + 1
        noun_end = noun_start + len(noun)
    else:
        noun_start = pair_start
        noun_end = pair_start + len(noun)
        connector = " that was "
        adj_start = noun_end + len(connector)
        adj_end = adj_start + len(adjective)

    if condition == 4:
        if data_type == 1:
            return _tokens_in_char_range(offset_mapping, adj_start, adj_end)
        else:
            connector_start = noun_end + 1
            return _tokens_in_char_range(offset_mapping, connector_start, adj_end)

    if condition == 5:
        return _tokens_in_char_range(offset_mapping, noun_start, noun_end)


def get_context_key(data_type: int, location: str) -> str:
    return f"data_type_{data_type}_location_{location}"
