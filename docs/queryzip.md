# QueryZip+

QueryZip+ combines KVzip+ reconstruction importance with attention from a
trailing question window. It is a query-aware, training-free extension of
[KVzip](https://arxiv.org/abs/2505.23416), using the KVzip+ normalization described
in [KVzap](https://arxiv.org/abs/2601.07891). These are references for the base
methods, not a published QueryZip paper.

## Algorithm

1. Prefill the document **with the question appended** and capture the final
   `query_window=64` query states at each layer, including RoPE and model-specific
   query normalization.
2. Run the existing KVzip+ context-reconstruction scoring passes.
3. Compute non-causal attention from the captured window to the original cache.
   Take the maximum over query tokens, average across GQA groups, and apply
   width-5 average pooling over cache positions.
4. Fuse scores globally across layers, heads, and tokens:
   `score = norm(reconstruction) + 0.5 * norm(query)`.
   The default `rank` mode uses ordinal percentile ranks with stable ties in
   flattened cache order. `amax` divides each map by its global maximum.
5. Protect the first four sink tokens and the trailing window, then apply
   KVzip's existing budgeted eviction. There is no additional hard query reserve.

Both fusion modes accumulate scores in float32. Rank normalization no longer
casts integer ranks to BF16/FP16 before division. Query states own only their
window storage, and grouped attention avoids explicitly duplicating all keys.
These numerical changes mean the experimental checkout's scores must be
re-evaluated; they are not submission results for this implementation.

The window is a heuristic, not a question parser: shorter questions include
some document tokens, while questions longer than the window use only their
tail. Increase `query_window` explicitly if needed. Sinks and the window count
toward the retained budget; an insufficient budget raises `ValueError`.

## Python usage

```python
from transformers import pipeline
from kvpress import QueryZipPress

pipe = pipeline(
    "kv-press-text-generation",
    model="Qwen/Qwen3-8B",
    device="cuda:0",
    dtype="bfloat16",
    model_kwargs={"attn_implementation": "flash_attention_2"},
)
press = QueryZipPress(compression_ratio=0.875)
document = "..."
question = "\nWhich events are described in this document?"
answer = pipe(document + question, question="", press=press)["answer"]
```

The question must be present during prefill. Passing it separately as
`pipe(document, question=question, ...)` would make it unavailable during
compression. Recompress for each question; this is not a reusable query-agnostic
document cache. Avoid truncating the appended question with `max_context_length`.

`QueryZipPress(query_score_mode="amax")` selects the alternative fusion mode.
`query_blend=0` is a protection-only ablation, not exactly vanilla KVzip+.
`kvzip_plus_normalization=False` disables the base scorer's plus normalization.

## Evaluation on one GPU

From the repository root, install `uv sync --extra eval --extra flash-attn`.
Then run from `evaluation/`:

```bash
uv run python evaluate.py \
  --model Qwen/Qwen3-8B --dataset ruler --data_dir 4096 \
  --press_name queryzip_plus --query_aware \
  --compression_ratio 0.875 --fraction 1.0 \
  --device cuda:0 --output_dir ./results_queryzip
```

Registry entries are `queryzip_plus` (rank) and `queryzip_plus_amax` (amax).
The evaluator rejects these entries without `query_aware=true`. It appends the
question before prefill using the existing query-aware protocol. For a complete
curve, run ratios `0.25`, `0.50`, `0.75`, and `0.875` sequentially, then repeat
with `meta-llama/Llama-3.1-8B-Instruct`. Save each run's configuration, metrics,
and predictions. Use `kvzip_plus --query_aware` as the direct fusion control,
alongside query-agnostic baselines and `no_press`.

## Limits

The implementation supports one dense prefill with batch size one. It inherits
KVzip's model/backend restrictions, including no eager attention or Gemma3.
It does not support incremental/chunked prefill or decoding compression.
At zero compression the press is a no-op, with no tokenizer loading or scoring.

KVzip uses masked eviction in kvpress: the nominal retained budget describes
active KV pairs, **not physically reduced cache allocation**. Reconstruction
also requires extra forward passes. Report measured memory and latency
separately from accuracy; do not infer speed or memory gains from the ratio.
