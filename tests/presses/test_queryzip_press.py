# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress import QueryZipPress
from kvpress.presses.kvzip_press import KVzipPress


class TinyTokenizer:
    """Only reconstruction prompt tokenization is needed; never download weights."""

    chat_template = None

    def encode(self, text, **kwargs):
        return torch.tensor([[ord(c) % 32 for c in text]], dtype=torch.long)


@pytest.fixture(params=["llama", "qwen3"])
def model(request, monkeypatch):
    torch.manual_seed(7)
    config_class, model_class = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[request.param]
    config = config_class(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    config._attn_implementation = "sdpa"
    monkeypatch.setattr(
        "kvpress.presses.kvzip_press.AutoTokenizer.from_pretrained", lambda *args, **kwargs: TinyTokenizer()
    )
    return model_class(config).eval()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_window": 0},
        {"query_window": 1.5},
        {"query_blend": -1},
        {"query_blend": float("nan")},
        {"query_kernel_size": 0},
        {"query_kernel_size": 2},
        {"query_score_mode": "unknown"},
        {"n_sink": -1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        QueryZipPress(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rank_precision_and_ties(dtype):
    # More than 65504 entries used to overflow FP16 ranks before division.
    scores = torch.zeros(70000, dtype=dtype)
    result = QueryZipPress()._normalize_scores(scores)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert result[0] == 0 and result[-1] == 1
    assert (result[1:] > result[:-1]).all()


def test_amax_zero_scores():
    result = QueryZipPress(query_score_mode="amax")._normalize_scores(torch.zeros(8, dtype=torch.float16))
    assert torch.equal(result, torch.zeros(8))


@pytest.mark.parametrize("mode", ["rank", "amax"])
@pytest.mark.parametrize("layerwise", [False, True])
@torch.no_grad()
def test_prefill_reconstruction_budget_and_reuse(model, mode, layerwise):
    press = QueryZipPress(compression_ratio=0.5, query_window=4, query_score_mode=mode, layerwise=layerwise)
    original_forward = model.model.forward
    for _ in range(2):
        cache = DynamicCache()
        ids = torch.randint(0, 32, (1, 40))
        with press(model):
            model(input_ids=ids, past_key_values=cache)
            captured = {i: q.clone() for i, q in press._query_states.items()}
            assert len(captured) == 2
            assert all(q.shape == (1, 4, 4, 8) for q in captured.values())
        assert model.model.forward == original_forward
        assert not press._query_states and press.score_val is None and press._cache is None
        assert cache.get_seq_length() == 40  # KVzip masks rather than physically pruning.
        total_evicted = 0
        for layer in model.model.layers:
            assert not layer.self_attn._forward_hooks
            indices = layer.self_attn.masked_key_indices
            total_evicted += indices[2].numel()
            assert ((indices[2] >= 4) & (indices[2] < 36)).all()
        assert total_evicted == int(2 * 2 * 40 * 0.5)
        # The actual attention patch must consume the eviction indices during decoding.
        output = model(input_ids=torch.tensor([[3]]), past_key_values=cache)
        assert torch.isfinite(output.logits).all()


@torch.no_grad()
def test_grouped_query_scores_match_explicit_attention(model):
    press = QueryZipPress(query_window=4, query_kernel_size=1)
    cache = DynamicCache()
    hooks = [
        layer.self_attn.register_forward_hook(press._capture_query, with_kwargs=True) for layer in model.model.layers
    ]
    try:
        model(input_ids=torch.randint(0, 32, (1, 20)), past_key_values=cache)
    finally:
        for hook in hooks:
            hook.remove()
    press._cache = cache
    press.score_val = torch.zeros(2, 1, 2, 20)
    result = press._compute_query_scores(model)
    for i in range(2):
        queries = press._query_states[i]
        # Captured states must not retain full-context activation storage.
        assert queries.untyped_storage().nbytes() == queries.numel() * queries.element_size()
        keys = repeat_kv(cache.layers[i].keys, 2)
        weights = (queries @ keys.transpose(-2, -1) / 8**0.5).softmax(dim=-1)
        expected = weights.amax(dim=-2).reshape(1, 2, 2, 20).mean(dim=2)
        torch.testing.assert_close(result[i], expected)


@pytest.mark.parametrize("mode", ["rank", "amax"])
def test_query_signal_changes_eviction(mode):
    attention = SimpleNamespace(layer_idx=0, config=SimpleNamespace(_attn_implementation="sdpa"))
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    reconstruction = torch.tensor([[[[0.0, 0.9, 0.8, 0.7, 0.01, 0.0]]]])
    query = torch.tensor([[[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]])
    kept = []
    for blend in (0.0, 2.0):
        press = QueryZipPress(compression_ratio=0.5, n_sink=1, query_window=1, query_blend=blend, query_score_mode=mode)
        press.score_val = reconstruction.clone()
        with patch.object(press, "_compute_query_scores", return_value=query):
            press.compress_post(model)
        kept.append(set(range(6)) - set(attention.masked_key_indices[2].tolist()))
    assert kept[0] == {0, 1, 5}
    assert kept[1] == {0, 4, 5}


@pytest.mark.parametrize("layerwise", [False, True])
@torch.no_grad()
def test_impossible_budget_cleans_up(model, layerwise):
    press = QueryZipPress(compression_ratio=0.9, query_window=8, layerwise=layerwise)
    with pytest.raises(ValueError, match="budget cannot retain"):
        with press(model):
            model(input_ids=torch.randint(0, 32, (1, 20)), past_key_values=DynamicCache())
    assert not press._query_states and press._cache is None
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)


@pytest.mark.parametrize("phase", ["prefill", "reconstruction"])
def test_exception_restores_hooks_and_forward(model, phase):
    press = QueryZipPress(compression_ratio=0.5, query_window=4)
    original_forward = model.model.forward
    with patch.object(KVzipPress, "_perform_kvzip_compression", side_effect=RuntimeError("reconstruction")):
        with pytest.raises(RuntimeError, match=phase):
            with torch.no_grad(), press(model):
                model(input_ids=torch.randint(0, 32, (1, 40)), past_key_values=DynamicCache())
                if phase == "prefill":
                    raise RuntimeError("prefill")
    assert model.model.forward == original_forward
    assert not press._query_states and press._cache is None
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)


@torch.no_grad()
def test_zero_compression_is_exact_noop(model):
    ids = torch.randint(0, 32, (1, 20))
    expected = model(input_ids=ids, past_key_values=DynamicCache()).logits
    with patch("kvpress.presses.kvzip_press.AutoTokenizer.from_pretrained", side_effect=AssertionError("download")):
        with QueryZipPress()(model):
            actual = model(input_ids=ids, past_key_values=DynamicCache()).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
