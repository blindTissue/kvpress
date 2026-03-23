# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from transformers import DynamicCache, PreTrainedModel
from transformers.models.llama.modeling_llama import rotate_half


def _rerotate_cos_sin(x, inv_freq, selected_positions):
    """
    Compute cos/sin rotary embeddings needed to re-rotate pruned keys into
    contiguous positions. Ported from KeyRerotationPress._rerotate_cos_sin.
    """
    bsz, num_key_value_heads, n_kept = selected_positions.shape
    device = selected_positions.device
    device_type = x.device.type
    dtype = x.dtype

    idx = torch.arange(0, n_kept, device=device).unsqueeze(0)
    inv_freq = inv_freq[None, None, :, None].float().expand(bsz, num_key_value_heads, -1, 1)
    idx = idx[:, None, :].float().expand(bsz, num_key_value_heads, n_kept)

    delta_pos = idx - selected_positions
    delta_pos = delta_pos.unsqueeze(2)

    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        freqs = delta_pos.float() * inv_freq.float()
        freqs = freqs.transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().contiguous()
        sin = emb.sin().contiguous()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _rerotate_keys(
    inv_freq: torch.Tensor, head_dim: int, indices: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    """
    Re-rotate keys so kept positions get contiguous RoPE embeddings.
    Ported from KeyRerotationPress.rerotate_keys.

    Parameters
    ----------
    inv_freq : torch.Tensor
        Rotary embedding inverse frequencies from ``language_model.rotary_emb.inv_freq``.
    head_dim : int
        Dimension of each attention head.
    indices : torch.Tensor
        Shape (bsz, num_heads, n_kept) -- indices of kept positions.
    keys : torch.Tensor
        Shape (bsz, num_heads, seq_len, head_dim).
    """
    new_cos, new_sin = _rerotate_cos_sin(keys, inv_freq, indices)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    keys = keys.gather(2, indices_expanded).contiguous()
    return (keys * new_cos) + (rotate_half(keys) * new_sin)


def evict_from_cache(
    cache: DynamicCache,
    positions_to_evict: list[int],
    rerotate: bool = False,
    model: PreTrainedModel = None,
) -> tuple[DynamicCache, int]:
    """
    Evict specific token positions from all layers of a DynamicCache.

    Parameters
    ----------
    cache : DynamicCache
        The KV cache to modify. Modified in-place.
    positions_to_evict : list[int]
        Token indices (along the sequence dimension) to remove.
    rerotate : bool
        If True, re-rotate keys so remaining positions get contiguous RoPE
        embeddings (0..n_kept-1). Requires `model`.
    model : PreTrainedModel, optional
        Required when rerotate=True.

    Returns
    -------
    tuple[DynamicCache, int]
        The modified cache and the original sequence length before eviction.
    """
    if not positions_to_evict:
        return cache, cache.get_seq_length()

    if rerotate and model is None:
        raise ValueError("model is required when rerotate=True")

    seq_len = cache.get_seq_length()
    original_len = seq_len

    evict_set = set(positions_to_evict)
    keep_indices = [i for i in range(seq_len) if i not in evict_set]
    device = cache.layers[0].keys.device
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=device)

    if rerotate:
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )
        inv_freq = language_model.rotary_emb.inv_freq

    for layer_idx in range(len(cache)):
        layer = cache.layers[layer_idx]
        keys = layer.keys
        values = layer.values
        bsz, num_heads, _, head_dim = keys.shape

        if rerotate:
            indices_for_rerotate = keep_tensor.unsqueeze(0).unsqueeze(0).expand(
                bsz, num_heads, -1
            )
            keys = _rerotate_keys(inv_freq, head_dim, indices_for_rerotate, keys)
            gather_idx = keep_tensor.view(1, 1, -1, 1).expand(bsz, num_heads, -1, head_dim)
            values = values.gather(2, gather_idx).contiguous()
        else:
            gather_idx = keep_tensor.view(1, 1, -1, 1).expand(bsz, num_heads, -1, head_dim)
            keys = keys.gather(2, gather_idx).contiguous()
            values = values.gather(2, gather_idx).contiguous()

        layer.keys = keys
        layer.values = values

    return cache, original_len


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """
    Deep-copy a DynamicCache so the original is not modified.

    Parameters
    ----------
    cache : DynamicCache
        The cache to copy.

    Returns
    -------
    DynamicCache
        A new DynamicCache with cloned key/value tensors.
    """
    new_cache = DynamicCache()
    for layer_idx in range(len(cache)):
        layer = cache.layers[layer_idx]
        new_cache.update(
            layer.keys.clone(),
            layer.values.clone(),
            layer_idx,
        )
    return new_cache
