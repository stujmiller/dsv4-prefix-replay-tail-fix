# fix-dsv4-prefix-replay-tail

A vLLM prefix-cache fix for DeepSeek V4 Flash with DSpark speculative decoding.
It targets vLLM 0.28.1rc1: the `0.29-b12x` DGX Spark images.

## The bug

About 1 in 4 requests leave nothing reusable in the prefix cache. When a prompt
ends 1 to 64 tokens past a 256-token boundary, an exact replay of it, or the
next turn of the same conversation, gets **0 cached tokens** and a full cold
prefill. The next turn falls back to the last turn that did cache, and
consecutive bad turns compound.

It is deterministic and has nothing to do with eviction, so it also makes
cache-pressure style retention benchmarks stop early and under-report.

**Cause.** `prefix_cache_retention_interval` defaults to `0`, so sliding-window
groups keep only the tail at the replay boundary. With DSpark, the 64-token
sliding-window group is an EAGLE group. Its hit run needs a full 64-token peek
block starting on the aligned 256-token boundary. When the prompt ends 1 to 64
tokens past that boundary, the block is never full, so the only tail retained
is unreachable. The hybrid hit is the minimum over groups, so it collapses to 0.

This is the same defect class as patch `04-boundfix` in
[co-l/ds4-prefix-cache-fixes](https://github.com/co-l/ds4-prefix-cache-fixes),
which was written for vLLM 0.26 and does not apply to the 0.28.1 cache manager.
This is a separate implementation for 0.28.1.

## The fix

Every KV cache manager also retains the tail at the last *reachable* boundary
(`num_prompt - 1 - slack`, where slack is the EAGLE sliding-window block size).
It changes two files: `v1/core/single_type_kv_cache_manager.py` and
`v1/core/kv_cache_coordinator.py`.

- It only adds retained blocks; it never removes any.
- It is a no-op when there is no EAGLE sliding-window group.
- It is Python-only, with no rebuild.

## Using it

The mod follows the spark-vllm-docker `--apply-mod` format: a directory with
`run.sh` plus a patch. It is idempotent.

```bash
./launch-cluster.sh ... --apply-mod /path/to/mods/fix-dsv4-prefix-replay-tail ...
```

On startup, each node should log `[fix-dsv4-prefix-replay-tail] done`.

## Results

Measured on 2x DGX Spark, TP=2, with DSpark k=6, MNBT 4096 and the default
retention interval. Cached tokens were read from vLLM's `prefix_cache_hits_total`.

| | without fix | with fix |
|---|---|---|
| exact re-send, prompt 1-64 tokens past a 256 boundary | 0 cached (7/7 cold) | hit, one block back (16/16) |
| next turn after such a prompt | 0 cached (7/7 cold) | reuses turn 1 (4/4) |
| [co-l/cache-pressure](https://github.com/co-l/cache-pressure) sanity, 3 contexts | 0/3 | 3/3 |
| cache-pressure retention, 8K contexts | 3 contexts, 0.82% | 258 contexts, 2.07M tokens, 77.6% |
| needle-in-haystack 50K-450K | 5/5 | 5/5 |

Offline, the image's own KV cache manager was driven across all 256 prompt end
offsets at 8K, 65K and 262K prompt lengths, with MNBT 4096 and 2048. Zero-hit
cases went from 192/768 to 0/768.

## Not included

co-l's duplicate-block dedupe (`03-dedupe`) is not part of this fix.
