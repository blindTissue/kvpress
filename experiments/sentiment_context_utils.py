# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

_CREATION_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "custom_dataset_creation"
    / "sentiment_dataset_creation.py"
)
_spec = importlib.util.spec_from_file_location("sentiment_dataset_creation", _CREATION_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
create_data = _mod.create_data


def create_context_without_target(
    sample: dict,
    data_type: int,
    location: str,
) -> str:
    """
    Rebuild the context string with the target pair removed.

    Used for condition 2 (no-information baseline) where the target pair is
    absent from the context entirely.
    """
    pair_list = sample["pair_list"]
    target_index = sample["target_index"]
    reduced_pairs = [p for i, p in enumerate(pair_list) if i != target_index]

    adj_first = data_type == 1
    location_first = location == "first"

    return create_data(reduced_pairs, adj_first=adj_first, location_first=location_first)
