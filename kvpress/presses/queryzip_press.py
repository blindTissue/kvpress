# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states


@dataclass
class QueryZipPress(KVzipPress):
    """QueryZip+: fuse KVzip+ reconstruction scores with trailing-query attention.

    Append the question to the context BEFORE entering this press (or evaluate
    with ``query_aware=true``). The trailing ``query_window`` prefill tokens are
    an observation window, not an automatically detected question boundary.
    A separately supplied pipeline ``question`` is only seen after compression.

    Scores are fused globally as ``norm(reconstruction) + query_blend * norm(query)``.
    Query scores use maximum attention over the window, mean over GQA groups,
    and average pooling. As in the experimental method, this scoring attention
    is non-causal over the original prefill cache. Sinks and the trailing window
    are protected within the requested budget; impossible budgets raise an error.

    Like KVzipPress, this supports a single dense prefill, batch size one, and
    uses masked eviction rather than physically shrinking cache storage. It
    requires reconstruction passes and is not a decoding press.

    Parameters
    ----------
    query_window : int, default=64
        Number of trailing prefill tokens used to score and protect the query.
    query_blend : float, default=0.5
        Nonnegative weight of query scores. Zero disables query scoring but
        still protects sinks and the window (a protection-only ablation).
    query_kernel_size : int, default=5
        Positive odd width of average pooling over query scores.
    query_score_mode : str, default="rank"
        Global ordinal percentile ranks ("rank") or maximum normalization
        ("amax"). Rank ties follow flattened cache order deterministically.
    kvzip_plus_normalization : bool, default=True
        Enable KVzip+ reconstruction normalization. Disable for a KVzip ablation.
    """

    kvzip_plus_normalization: bool = True
    query_window: int = 64
    query_blend: float = 0.5
    query_kernel_size: int = 5
    query_score_mode: str = "rank"
    _query_states: dict[int, torch.Tensor] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.query_window, int) or self.query_window < 1:
            raise ValueError("query_window must be a positive integer")
        if not math.isfinite(self.query_blend) or self.query_blend < 0:
            raise ValueError("query_blend must be finite and nonnegative")
        if not isinstance(self.query_kernel_size, int) or self.query_kernel_size < 1 or self.query_kernel_size % 2 == 0:
            raise ValueError("query_kernel_size must be a positive odd integer")
        if self.query_score_mode not in ("rank", "amax"):
            raise ValueError("query_score_mode must be 'rank' or 'amax'")
        if not isinstance(self.n_sink, int) or self.n_sink < 0:
            raise ValueError("n_sink must be a nonnegative integer")

    @torch.no_grad()
    def _capture_query(self, module: nn.Module, args, kwargs: dict, output):
        layer_idx = int(module.layer_idx)
        if layer_idx in self._query_states:
            raise ValueError("QueryZipPress requires a single dense prefill per context manager")
        hidden_states = kwargs["hidden_states"]
        if hidden_states.shape[0] != 1:
            raise ValueError("QueryZipPress supports batch size one")
        window = min(self.query_window, hidden_states.shape[1])
        if window == 0:
            raise ValueError("QueryZipPress requires a nonempty prefill")
        # Own only the window storage; detach alone would retain the full prefill.
        hidden_states = hidden_states[:, -window:].detach().clone()
        cos, sin = kwargs["position_embeddings"]
        queries = get_prerope_query_states(module, hidden_states)
        cos, sin = cos[:, -window:].unsqueeze(1), sin[:, -window:].unsqueeze(1)
        self._query_states[layer_idx] = (queries * cos + rotate_half(queries) * sin).detach()

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        self._query_states.clear()
        if self.compression_ratio == 0:
            yield
            return
        hooks = []
        try:
            with super().__call__(model):
                try:
                    for layer in model.model.layers:
                        hooks.append(layer.self_attn.register_forward_hook(self._capture_query, with_kwargs=True))
                    yield
                finally:
                    # Reconstruction must not overwrite the original question states.
                    for hook in hooks:
                        hook.remove()
        finally:
            self._query_states.clear()

    def _compute_query_scores(self, model: PreTrainedModel) -> torch.Tensor:
        query_scores = torch.zeros_like(self.score_val, dtype=torch.float32)
        ctx_len = self.score_val.shape[-1]
        for layer in model.model.layers:
            module = layer.self_attn
            layer_idx = int(module.layer_idx)
            if layer_idx not in self._query_states:
                raise RuntimeError(f"Missing prefill query states for layer {layer_idx}")
            queries = self._query_states[layer_idx]
            keys, _ = extract_keys_and_values(self._cache, layer_idx)
            keys = keys[:, :, :ctx_len]
            bsz, num_kv_heads, _, head_dim = keys.shape
            num_groups = queries.shape[1] // num_kv_heads
            queries = queries.reshape(bsz, num_kv_heads, num_groups, -1, head_dim)
            # Broadcast KV heads across GQA groups without repeating the full keys.
            weights = torch.matmul(queries, keys.unsqueeze(2).transpose(-2, -1)) / math.sqrt(head_dim)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
            scores = weights.amax(dim=-2).mean(dim=2)
            if 1 < self.query_kernel_size <= ctx_len:
                scores = F.avg_pool1d(scores, self.query_kernel_size, stride=1, padding=self.query_kernel_size // 2)
            query_scores[layer_idx] = scores.to(query_scores.device)
        return query_scores

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        if self.query_score_mode == "amax":
            return scores / scores.amax().clamp_min(1e-8)
        flat = scores.flatten()
        order = flat.argsort(stable=True)
        ranks = torch.empty_like(flat)
        ranks[order] = torch.arange(flat.numel(), device=flat.device, dtype=torch.float32)
        return (ranks / max(flat.numel() - 1, 1)).view_as(scores)

    @torch.no_grad()
    def compress_post(self, model: PreTrainedModel):
        if self.compression_ratio == 0:
            return
        ctx_len = self.score_val.shape[-1]
        window = min(self.query_window, ctx_len)
        protected_per_head = min(ctx_len, self.n_sink + window)
        # Match KVzip's integer rounding, including its layerwise option.
        budget_size = self.score_val[0].numel() if self.layerwise else self.score_val.numel()
        n_kept = budget_size - int(budget_size * self.compression_ratio)
        n_protected = budget_size // ctx_len * protected_per_head
        if n_kept < n_protected:
            raise ValueError(
                "QueryZipPress budget cannot retain the sinks and query window; reduce compression or window"
            )
        if self.query_blend > 0:
            query_scores = self._compute_query_scores(model)
            self.score_val = self._normalize_scores(self.score_val) + self.query_blend * self._normalize_scores(
                query_scores
            )
        else:
            self.score_val = self.score_val.float()
        boost = self.score_val.amax() + 1.0
        self.score_val[..., : self.n_sink] = boost
        self.score_val[..., -window:] = boost
        super().compress_post(model)
