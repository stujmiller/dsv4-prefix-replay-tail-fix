# fix-dsv4-prefix-replay-tail

A small patch for vLLM's prefix cache. It targets **DeepSeek V4 Flash** served with
**DSpark speculative decoding** on **vLLM 0.28.1rc1**, the `0.29-b12x` DGX Spark
images.

**What it fixes.** About 1 in 4 requests leave nothing reusable in the prefix
cache. If a prompt ends 1 to 64 tokens past a 256-token boundary, re-sending it,
or sending the next turn of that conversation, gets **0 cached tokens**. You pay
a full cold prefill: minutes at a few hundred thousand tokens. After the fix,
those requests hit the cache, one 256-token block short of a full hit.

**What it changes.** Two Python files inside vLLM
(`v1/core/single_type_kv_cache_manager.py` and `v1/core/kv_cache_coordinator.py`),
27 added lines, no rebuild. It only makes the cache keep a few extra blocks, and it
never removes anything. See [How the fix works](#how-the-fix-works).

## Does this affect you?

All of these must be true:

- vLLM **0.28.1rc1**, e.g. `dicksondickson/vllm_spark_dsv4:0.29-b12x` or
  `0rand/vllm_spark_dsv4-0.29-b12x`. `run.sh` refuses to apply if the two files
  don't match what it expects.
- **DeepSeek V4 Flash** (any variant) with **`--enable-prefix-caching`**.
- **DSpark** speculative decoding (`"method": "dspark"`).
- `prefix_cache_retention_interval` left at its default of `0`
  (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` not set).

To check a running endpoint, run [`check_deadzone.py`](#3-check-that-it-works)
against it. `RESULT: dead zone PRESENT` means you are affected.

## How to apply

Apply it **on every node** (head and workers), **before `vllm serve` starts**. vLLM
loads these files at startup, so patching a running server does nothing until
vLLM restarts. A recreated container starts unpatched, so it must be re-applied
on every container start. The launcher methods below do that for you.

The image needs the `patch` command. The `0.29-b12x` images have it.

### Option A: spark-vllm-docker launcher (`--apply-mod`)

Copy `mods/fix-dsv4-prefix-replay-tail/` onto the head node and pass it to the
launcher. The launcher copies it to the workers and runs it in every container
before starting vLLM:

```bash
./launch-cluster.sh ... --apply-mod /path/to/mods/fix-dsv4-prefix-replay-tail ...
```

If you use 0rand's `start-cluster.sh`, add it next to the existing mods in
`FIX_MOD_ARGS`:

```bash
FIX_MOD_ARGS+=(--apply-mod "$DIR/mods/fix-dsv4-prefix-replay-tail")
```

### Option B: by hand in a container

On every node, with the container up but before vLLM starts (or restart vLLM
afterwards):

```bash
docker cp mods/fix-dsv4-prefix-replay-tail <container>:/tmp/
docker exec <container> bash /tmp/fix-dsv4-prefix-replay-tail/run.sh
```

### Option C: bake it into your own image

```dockerfile
COPY mods/fix-dsv4-prefix-replay-tail /tmp/fix-dsv4-prefix-replay-tail
RUN bash /tmp/fix-dsv4-prefix-replay-tail/run.sh
```

### 1. Check that it applied

Each node's startup output should show:

```
[fix-dsv4-prefix-replay-tail] applying patch to .../vllm
[fix-dsv4-prefix-replay-tail] done
```

It prints `already applied - skipping` if the files are already patched. It prints
`ERROR: cannot apply` and exits non-zero if your vLLM files differ, which usually
means a different version. In that case nothing is changed.

To confirm inside a running container (run on every node):

```bash
docker exec <container> sh -c 'grep -c fix-dsv4-prefix-replay-tail $(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/v1/core/kv_cache_coordinator.py'
# 1 = patched, 0 = not patched
```

### 2. Restart vLLM

If you patched by hand into a container where vLLM was already running, restart
vLLM. Options A and C patch before start, so this step is not needed.

### 3. Check that it works

`check_deadzone.py` sends about 22 prompts of ~8K tokens, each twice, and reads
vLLM's own cache counters. It tolerates concurrent traffic: a sample is only
accepted when its cache-query delta equals its own prompt-token count, so a
request that touches the cache in the same window is what gets discarded. Run it
while traffic is light for the cleanest result:

```bash
pip install requests
python3 check_deadzone.py http://HOST:8000
```

```
prompts ending 1-64 past a 256 boundary: 5 hit, 0 got 0 cached
all other prompts:                       17 hit, 0 got 0 cached
RESULT: no dead zone (fix active, or this setup is not affected)
```

Exit code 0 means no dead zone, 1 means dead zone present, and 2 means
inconclusive (every dead-zone-length sample's cache window was contaminated by
other traffic; re-run it).

## How to remove it

Take the mod out (drop the `--apply-mod` line, or rebuild without it) and restart
the containers. The patch only exists in the running container's files.

## The bug

**Symptom.** A request whose prompt ends 1 to 64 tokens past a 256-token boundary
leaves nothing reusable. Its exact replay and its follow-up turn get 0 cached
tokens. The follow-up falls back to the last turn that did cache, and consecutive
bad turns compound. It is deterministic and has nothing to do with eviction. That
also makes cache-pressure style retention benchmarks stop early and report far
too little retention.

**Cause.** DeepSeek V4 in vLLM uses several KV cache groups per request. A prefix
hit must be valid in all of them: the hit is the minimum over groups. One of them
is a 64-token sliding-window group, which DSpark also uses, so vLLM treats it as an
EAGLE group. For an EAGLE group, a hit needs a *complete* 64-token block starting
exactly on a 256-token boundary.

On this vLLM version `prefix_cache_retention_interval` defaults to `0`. Sliding-window
groups then keep only one tail per request: the one at the prompt's last 256-token
boundary. If the prompt ends 1 to 64 tokens past that boundary, the 64-token block
there is never complete. The one tail that was kept can never be used, and nothing
else was kept, so the hit is 0.

This is the same defect class as patch `04-boundfix` in
[co-l/ds4-prefix-cache-fixes](https://github.com/co-l/ds4-prefix-cache-fixes).
That patch was written for vLLM 0.26 and does not apply to the rewritten 0.28.1
cache manager; this is a separate implementation.

## How the fix works

- `kv_cache_coordinator.py`: at startup, compute a "slack" equal to the block size
  of the EAGLE sliding-window group (64 here), and give the same value to every
  cache group. It is 0 when there is no such group.
- `single_type_kv_cache_manager.py`: when deciding which tail blocks to keep for a
  request, also keep the tail at `num_prompt - 1 - slack`. That is the last
  boundary a lookup can actually use. Every group keeps that same boundary,
  because the hit is the minimum over groups.

Nothing is removed and lookup logic is unchanged. The cost is a handful of extra
small blocks per request.

## Results

Measured on 2x DGX Spark, TP=2, `0.29-b12x` image, with the default retention
interval. Cached tokens were read from vLLM's `prefix_cache_hits_total`. Cache
tests used [co-l/cache-pressure](https://github.com/co-l/cache-pressure).

### Config A: marlin MoE, DSpark k=5, MNBT 2048, max-num-seqs 12

This is the serving config from
[oselivanov/ollie-gb10-serving-stacks](https://github.com/oselivanov/ollie-gb10-serving-stacks),
where the cache problem was first reported. Both columns are the same pair and the
same config; the only difference is the fix.

| | without fix | with fix |
|---|---|---|
| exact re-send, prompt 1-64 tokens past a 256 boundary | 0 cached (6/6 cold) | hit, one block back (11/11) |
| exact re-send, zero-hit results overall | 6/22 | 0/44 |
| next turn after a dead-zone turn 1 | not measured | reuses turn 1 (8/8) |
| cache-pressure sanity, 3 contexts | 0/3 | 3/3 |
| cache-pressure retention, 8K contexts | 3/374 contexts, 0.82% | 252/411 contexts, 2.02M tokens, 62.2%* |
| needle-in-haystack 50K-450K | 5/5 | 5/5 |
| needle TTFT 50K / 100K / 200K / 300K / 450K | 26 / 55 / 121 / 205 / 354s | 27 / 55 / 126 / 211 / 348s |

\* Advertised capacity differed between boots (2.95M without the fix, 3.25M with
it), so compare retained tokens rather than percentages.

### Config B: b12x MoE, DSpark k=6, MNBT 4096, max-num-seqs 4

"Without fix" was measured on a second pair with an identical image and config.

| | without fix | with fix |
|---|---|---|
| exact re-send, prompt 1-64 tokens past a 256 boundary | 0 cached (3/3 cold) | hit, one block back (16/16) |
| exact re-send, zero-hit results overall | 3/22 | 0/44 |
| next turn after a dead-zone turn 1 | 0 cached (7/7 cold) | reuses turn 1 (4/4) |
| cache-pressure sanity, 3 contexts | not measured | 3/3 |
| cache-pressure retention, 8K contexts | not measured | 258/339 contexts, 2.07M tokens, 77.6% |
| needle-in-haystack 50K-450K | not measured | 5/5 |

Neither config logged any Traceback, EngineDeadError or NVRM error with the fix.

### Offline verification

The image's own KV cache manager code was driven directly. With a DeepSeek V4 group
layout, it reproduced the live measurements exactly, including specific prompt
lengths. It was then swept across all 256 prompt end offsets at 8K, 65K and 262K
prompt lengths, with MNBT 4096 and 2048 and speculative lookahead for k=5 and k=6.
Zero-hit cases went from **192/768 to 0/768** in every combination.

## Caveats

- **Scope.** Tested only on vLLM 0.28.1rc1 (`0.29-b12x`) with DSpark on 2x DGX
  Spark. MTP, other drafters, other vLLM versions and other hardware are untested.
- **No long soak.** It has not run for days under real mixed traffic. The live
  probes were sequential; many concurrent sessions were not specifically tested.
- **Prompt sizes.** The live dead-zone probes used ~8K-token prompts. Longer
  prompts (65K and 262K) were covered offline only. The live needle test confirms
  prompts up to 450K still work, but it does not probe the dead zone.
- **One ambiguous miss.** In Config A's cache-pressure run, the first miss was a
  dead-zone-length context. All 77 other dead-zone-length contexts in the retained
  set hit, so it is most likely genuine eviction, but one run cannot strictly
  prove it.
- **Not all of co-l's work.** This fixes the dead zone only. co-l's full patch set
  (including `03-dedupe`) reported about 147% retention on a different vLLM
  version; these numbers are not comparable to that.
- **The check script's busy-detection tolerates concurrent traffic** (each sample
  must have a clean cache-query window, rather than requiring the endpoint to be
  completely idle). It has been exercised on both an unpatched endpoint (reports
  `dead zone PRESENT`) and a patched one (reports `no dead zone`).

## Files

- `mods/fix-dsv4-prefix-replay-tail/run.sh`: applies the patch; idempotent and safe
  to re-run.
- `mods/fix-dsv4-prefix-replay-tail/dsv4-prefix-replay-tail.patch`: the change
  itself.
- `check_deadzone.py`: checks an endpoint for the dead zone.
