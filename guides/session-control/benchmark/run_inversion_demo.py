#!/usr/bin/env python3
"""Synthetic inversion demo: manufacture the one scenario where close_session
must beat LRU, and measure it directly.

Per arm (close / noclose), on ONE flushed sglang pod via port-forward:
  1. RESIDENT FILL - open resident sessions (~150k tokens each, distinct random
     prefixes) until the KV pool is ~88% full, then leave them idle.
  2. CHURN - run short-lived sessions (~80k each) sequentially. Arm close POSTs
     /close_session after each finishes; arm noclose leaves them open.
  3. PROBE - every resident sends its full prefix + one tiny turn; record
     cached_tokens/prompt_tokens (survival) and wall latency (~TTFT at 8 tokens).

Prediction: in noclose the referenced-but-fresh churn blocks outrank the idle
residents in LRU order, so churn evicts residents; in close the churn blocks are
unreferenced the moment they die and get evicted first, sparing residents.

Usage: python3 run_inversion_demo.py [--churn 25] [--churn-tokens 80000]
Results: /tmp/inversion-demo/results.json (+ per-arm detail)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_close_session_bench as b  # port-forward / http / pod helpers

OUT = Path("/tmp/inversion-demo")
TARGET_FILL = 0.88          # resident fill fraction of the pool
RESIDENT_TOKENS = 150_000   # per-resident context target
PROBE_MAX_TOKENS = 8
FILL_MAX_TOKENS = 16


def words(seed: str, approx_tokens: int) -> str:
    """Random 4-8 letter pseudo-words; ~2 tokens/word for Qwen tokenizers.
    Distinct seeds -> distinct prefixes, so sessions share no radix prefix."""
    rng = random.Random(seed)
    n = int(approx_tokens / 2.0)
    return " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz")
                            for _ in range(rng.randint(4, 8))) for _ in range(n))


def chat(session_id: str, content: str, max_tokens: int, timeout: int = 600):
    t0 = time.perf_counter()
    st, r = b.http("POST", f"{b.BASE}/v1/chat/completions",
                   {"model": b.MODEL, "session_id": session_id, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": content}]}, timeout=timeout)
    lat = time.perf_counter() - t0
    if st != 200:
        raise RuntimeError(f"chat {session_id} -> HTTP {st}: {str(r)[:200]}")
    usage = r.get("usage", {})
    det = usage.get("prompt_tokens_details") or {}
    return dict(latency=lat, prompt=usage.get("prompt_tokens", 0),
                cached=det.get("cached_tokens", 0))


def run_arm(arm: str, pool: int, churn_n: int, churn_tokens: int) -> dict:
    print(f"\n===== arm {arm} =====")
    b.flush_cache()
    time.sleep(3)

    # 1. resident fill (sequential -> monotone LRU ages, oldest first)
    residents, total = [], 0
    i = 0
    while total + RESIDENT_TOKENS * 0.9 < pool * TARGET_FILL and i < 20:
        sid = f"resident-{arm}-{i}"
        text = words(sid, RESIDENT_TOKENS)
        u = chat(sid, text, FILL_MAX_TOKENS)
        residents.append(dict(sid=sid, text=text, fill_prompt=u["prompt"]))
        total += u["prompt"]
        print(f"  fill {sid}: {u['prompt']} tok in {u['latency']:.1f}s "
              f"(pool {total / pool * 100:.0f}%)")
        i += 1
    print(f"residents: {len(residents)}, total {total} tok = {total / pool * 100:.1f}% of pool")

    # 2. churn: short-lived sessions; close them only in the close arm
    closes_ok = 0
    for k in range(churn_n):
        sid = f"churn-{arm}-{k}"
        u = chat(sid, words(sid, churn_tokens), FILL_MAX_TOKENS)
        if arm == "close":
            st, _ = b.http("POST", f"{b.BASE}/close_session", {"session_id": sid})
            closes_ok += 1 if st == 200 else 0
        if (k + 1) % 5 == 0:
            print(f"  churn {k + 1}/{churn_n} ({u['prompt']} tok each)"
                  + (f", closes ok={closes_ok}" if arm == "close" else ""))

    # 3. probe every resident: same prefix + one tiny follow-up turn
    probes = []
    for r in residents:
        u = chat(r["sid"], r["text"] + "\n\nReply with the single word: ok",
                 PROBE_MAX_TOKENS, timeout=900)
        hit = u["cached"] / u["prompt"] if u["prompt"] else 0.0
        probes.append(dict(sid=r["sid"], hit=hit, latency=u["latency"],
                           cached=u["cached"], prompt=u["prompt"]))
        print(f"  probe {r['sid']}: hit={hit:.3f} latency={u['latency']:.1f}s")

    hits = [p["hit"] for p in probes]
    lats = [p["latency"] for p in probes]
    summary = dict(arm=arm, residents=len(residents), resident_tokens=total,
                   fill_frac=total / pool, churn_n=churn_n, closes_ok=closes_ok,
                   hit_mean=sum(hits) / len(hits), lat_mean=sum(lats) / len(lats),
                   survivors=sum(1 for h in hits if h > 0.5), probes=probes)
    print(f"[{arm}] resident hit mean={summary['hit_mean']:.3f} "
          f"survivors(hit>0.5)={summary['survivors']}/{len(residents)} "
          f"probe latency mean={summary['lat_mean']:.1f}s")
    return summary


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="")
    ap.add_argument("--churn", type=int, default=25)
    ap.add_argument("--churn-tokens", type=int, default=80_000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pod = b.pick_pod(args.pod)
    b.port_forward(pod)
    try:
        st, info = b.http("GET", f"{b.BASE}/get_server_info")
        assert st == 200 and info.get("enable_session_radix_cache") is True
        pool = int(info["max_total_num_tokens"])
        print(f"pod={pod} sglang={info.get('version')} pool={pool} session_radix=on")

        results = {arm: run_arm(arm, pool, args.churn, args.churn_tokens)
                   for arm in ("close", "noclose")}
        (OUT / "results.json").write_text(json.dumps(results, indent=1))

        print("\n===== INVERSION DEMO SUMMARY =====")
        for arm, s in results.items():
            print(f"{arm:8s} resident hit mean={s['hit_mean']:.3f} "
                  f"survivors={s['survivors']}/{s['residents']} "
                  f"probe latency mean={s['lat_mean']:.1f}s")
        print(f"details: {OUT}/results.json")
    finally:
        if b._pf_proc:
            b._pf_proc.terminate()


if __name__ == "__main__":
    main()
