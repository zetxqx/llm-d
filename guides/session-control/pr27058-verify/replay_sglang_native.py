#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Replay a captured agent trace against a SINGLE raw SGLang worker -- no Dynamo, no
router, no NATS. Tests sglang#27058 (session radix cache) directly through SGLang's
native API:

  - tag a request: POST /generate with session_params={"id": <session_id>}
  - release:       POST /close_session {"session_id": <session_id>}
  - (open_session is a no-op / implicit under --enable-session-radix-cache)

Same trace + same fidelity as the Dynamo-path replay (replay_session_trace.py), but
talks SGLang directly. Launch the server with run_sglang_native.sh, then:

  python replay_sglang_native.py golden-trace-12x4.jsonl --mode on  --speedup 4
  python replay_sglang_native.py golden-trace-12x4.jsonl --mode off --speedup 4

--mode on  -> session_params tag + /close_session per session (the feature)
--mode off -> identical workload, no session_params (plain radix baseline)

Watch :30000/metrics: sglang:token_usage, sglang:kv_available_tokens, and
sglang:evicted_tokens_total{cache_type="RadixCache"}. OFF saturates + the eviction
counter climbs; ON stays bounded and it stays ~0. Python stdlib only.
"""
import argparse, json, sys, threading, time, urllib.error, urllib.request, uuid
from collections import defaultdict


def synth_tokens(hashes, block_size, input_length, vocab=0):
    toks = []
    for h in hashes:
        tid = h & 0xFFFFFFFF
        if vocab:
            tid %= vocab  # keep ids in range for small models; deterministic -> prefixes still share
        toks.extend([tid] * block_size)
        if len(toks) >= input_length:
            return toks[:input_length]
    return toks[:input_length]


def load_trace(path):
    sessions = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line).get("event", {})
        if ev.get("event_type") != "request_end":
            continue
        ctx = ev.get("agent_context") or {}
        sid = ctx.get("session_id")
        req = ev.get("request") or {}
        rp = req.get("replay") or {}
        if not (sid and rp.get("input_sequence_hashes") and rp.get("input_length") and rp.get("trace_block_size")):
            continue
        sessions[sid].append({
            "sid": sid,
            "recv_ms": req.get("request_received_ms", ev.get("event_time_unix_ms", 0)),
            "end_ms": ev.get("event_time_unix_ms", req.get("request_received_ms", 0)),
            "in_len": rp["input_length"],
            "block": rp["trace_block_size"],
            "hashes": rp["input_sequence_hashes"],
            "out_tokens": max(1, req.get("output_tokens", 1)),
        })
    for turns in sessions.values():
        turns.sort(key=lambda t: t["recv_ms"])
    return sessions


def post(url, path, body, timeout_s=120):
    req = urllib.request.Request(
        f"{url}{path}", data=json.dumps(body).encode(), method="POST",
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.status, r.read()


def gen_body(turn, mode, vocab, sid_suffix=""):
    body = {
        "input_ids": synth_tokens(turn["hashes"], turn["block"], turn["in_len"], vocab),
        # ignore_eos: synthetic token ids decode to gibberish, so without it the
        # model EOSes early and the trace's output lengths are not reproduced.
        "sampling_params": {"max_new_tokens": turn["out_tokens"], "temperature": 0.0,
                            "ignore_eos": True},
        "stream": False,
    }
    if mode == "on":
        # v0.5.15+ released API: radix-native sessions use the TOP-LEVEL
        # session_id field (implicit open, /close_session releases).
        # session_params={"id": ...} now selects the old dedicated-slot
        # session controller and would bypass the radix tagging under test.
        body["session_id"] = turn["sid"] + sid_suffix
    return body


def run_session(turns, args, stats, t0):
    base = turns[0]["recv_ms"]
    sleep_for = t0 + (base - args.global_min_ms) / 1000.0 / args.speedup - time.monotonic()
    if sleep_for > 0:
        time.sleep(sleep_for)
    prev_end = None
    for i, turn in enumerate(turns):
        if prev_end is not None:
            gap = max(0.0, (turn["recv_ms"] - prev_end) / 1000.0 / args.speedup)
            if gap > 0:
                time.sleep(min(gap, args.max_gap_s))
        try:
            status, _ = post(args.url, "/generate", gen_body(turn, args.mode, args.vocab, args.sid_suffix))
            with stats["lock"]:
                stats["turns"] += 1
                stats["ok" if status == 200 else "err"] += 1
        except (urllib.error.URLError, OSError) as e:
            with stats["lock"]:
                stats["turns"] += 1; stats["err"] += 1
            print(f"[{turn['sid'][:8]}] turn {i} error: {e}", file=sys.stderr)
        prev_end = turn["end_ms"]
    if args.mode == "on":
        try:
            post(args.url, "/close_session", {"session_id": turns[-1]["sid"] + args.sid_suffix}, timeout_s=10)
            with stats["lock"]:
                stats["closes"] += 1
        except (urllib.error.URLError, OSError):
            pass  # release is best-effort


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--url", default="http://127.0.0.1:30000", help="SGLang server base URL")
    ap.add_argument("--mode", choices=["on", "off"], default="on")
    ap.add_argument("--speedup", type=float, default=4.0)
    ap.add_argument("--max-gap-s", type=float, default=2.0)
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=0, help="modulo token ids by this (set for small-vocab models; 0=raw)")
    ap.add_argument("--sid-suffix", default="", help="append to session ids; needed for reruns because closed session ids are tombstoned server-side and can never re-tag KV")
    args = ap.parse_args()

    sessions = load_trace(args.trace)
    if not sessions:
        print("no replayable request_end events", file=sys.stderr); sys.exit(1)
    if args.max_sessions:
        sessions = dict(list(sessions.items())[: args.max_sessions])
    args.global_min_ms = min(t["recv_ms"] for v in sessions.values() for t in v)

    print(f"[sglang-native] replaying {len(sessions)} sessions, mode={args.mode}, speedup={args.speedup}x -> {args.url}")
    stats = {"lock": threading.Lock(), "turns": 0, "ok": 0, "err": 0, "closes": 0}
    t0 = time.monotonic()
    ts = [threading.Thread(target=run_session, args=(v, args, stats, t0)) for v in sessions.values()]
    for t in ts: t.start()
    for t in ts: t.join()
    print(f"[sglang-native] done in {time.monotonic()-t0:.1f}s: {stats['turns']} turns "
          f"({stats['ok']} ok, {stats['err']} err), {stats['closes']} /close_session")


if __name__ == "__main__":
    main()
