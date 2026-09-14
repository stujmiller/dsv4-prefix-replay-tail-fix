#!/usr/bin/env python3
"""Check whether a vLLM endpoint has the prefix-cache dead zone.

Sends unique ~8K-token prompts twice each, across a range of lengths, and reads
vLLM's own /metrics counters (prefix_cache_hits_total) around the second send.
A prompt ending 1-64 tokens past a 256-token boundary gets 0 cached tokens when
the bug is present. With the fix it hits (one 256-token block back).

A sample is only accepted as clean when its cache-query delta equals its own
prompt-token count: if another request prefilled in the same window it would
add queries and break that equality. ``num_requests_running`` is read for
context but NOT used as a hard filter -- it flickers on any live endpoint
(including a cluster that is serving the agent itself) and would otherwise
discard every sample and report "inconclusive". On a busy endpoint the query
delta is the reliable isolation check.

Run it while nothing else is using the endpoint for best results; it still
works (with a verdict) under light concurrent traffic as long as each request's
cache window stays isolated.

Usage: python3 check_deadzone.py http://HOST:8000 [--model NAME] [--samples 22]
Needs: pip install requests
"""
import argparse
import random
import sys
import time

import requests

WORDS = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
         "liquor jugs. How vexingly quick daft zebras jump. Sphinx of black quartz, "
         "judge my vow.").split()


def body(chars, seed):
    rng = random.Random(seed)
    out, n = [], 0
    while n < chars:
        w = rng.choice(WORDS)
        out.append(w)
        n += len(w) + 1
    return " ".join(out)[:chars]


def counters(base):
    d = {}
    for line in requests.get(base + "/metrics", timeout=10).text.splitlines():
        for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                  "vllm:num_requests_running"):
            if line.startswith(k + "{"):
                d[k] = d.get(k, 0.0) + float(line.split()[-1])
    return d


def send(base, model, text):
    r = requests.post(base + "/v1/chat/completions", timeout=900, json={
        "model": model, "max_tokens": 1,
        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": text}]})
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", help="e.g. http://192.168.1.10:8000 (no /v1)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--samples", type=int, default=22)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    model = a.model or requests.get(base + "/v1/models", timeout=10).json()["data"][0]["id"]

    salt = random.randrange(10**6, 10**7)
    dead_zone = {"hit": 0, "zero": 0}
    other = {"hit": 0, "zero": 0}
    # Track how many clean samples each bucket collected, so we can tell a true
    # "no dead zone" (many clean, all hit) from "not enough data" (0 clean).
    dead_zone["samples"] = 0
    other["samples"] = 0
    print(f"model {model}")
    print(f"{'prompt':>7} {'past256':>7} {'cached':>7}")
    for i in range(a.samples):
        text = body(31800 + i * 1500 // max(a.samples - 1, 1), salt + i)
        p = send(base, model, text)
        before = counters(base)
        send(base, model, text)
        after = counters(base)
        queried = after.get("vllm:prefix_cache_queries_total", 0.0) - before.get("vllm:prefix_cache_queries_total", 0.0)
        cached = int(after.get("vllm:prefix_cache_hits_total", 0.0) - before.get("vllm:prefix_cache_hits_total", 0.0))
        busy = queried != p  # another request polluted this request's cache window
        past = p % 256
        note = "   (cache window contaminated by other traffic, ignored)" if busy else ""
        print(f"{p:>7} {past:>7} {cached:>7}{note}")
        if busy:
            continue
        bucket = dead_zone if 1 <= past <= 64 else other
        bucket["samples"] += 1
        bucket["zero" if cached == 0 else "hit"] += 1

    print()
    print(f"prompts ending 1-64 past a 256 boundary: {dead_zone['hit']} hit, {dead_zone['zero']} got 0 cached")
    print(f"all other prompts:                       {other['hit']} hit, {other['zero']} got 0 cached")
    if dead_zone["samples"] == 0:
        print("RESULT: inconclusive (no clean dead-zone samples; run with more --samples)")
        return 2
    if dead_zone["zero"]:
        print("RESULT: dead zone PRESENT (fix not active, or not effective on this setup)")
        return 1
    print("RESULT: no dead zone (fix active, or this setup is not affected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
