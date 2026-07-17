#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone, session-aware replay of a Dynamo agent trace (`dynamo.agent.trace.v1`).

Purpose
-------
Reproduce a real multi-agent pi run against a live Dynamo frontend WITHOUT pi, the
pi-dynamo-provider, Node, or any model API key. The agent's decisions are already
baked into the captured trace as token counts + prefix hashes + timing, so a replay
only has to reproduce the KV/token workload -- which is all the radix-native session
cache (sglang#27058 / dynamo#10214) cares about.

It is the missing piece the mocker replay path does not provide: it emits the
`nvext.session_control` open/close lifecycle that drives the feature, keyed on each
trace request's `agent_context.session_id`.

Fidelity
--------
- Tokens are synthesized exactly like the mocker (`lib/mocker/src/loadgen/trace.rs`):
  `token_id = hash & 0xFFFFFFFF`, each hash expanded to `trace_block_size` copies,
  truncated to `input_length`. Sent as an IntegerArray `prompt` to `/v1/completions`
  (pre-tokenized passthrough), so the worker sees identical block hashes across a
  session's turns and the radix cache hits the same way the real run did.
- Per session: turns run sequentially (closed-loop, mirroring a real subagent), the
  first turn carries `session_control.action=open`, later turns carry the bare
  `session_id`, and after the last turn a throwaway `max_tokens=1` request carries
  `action=close` -- exactly what pi-dynamo-provider does (dynamo-provider.ts).
- Sessions start staggered at their real relative arrival offsets and keep their real
  inter-turn think/tool gaps, so cross-session concurrency (the KV pressure) is
  reproduced. `--speedup N` compresses wall-clock gaps; `--mode off` drops
  session_control for the apples-to-apples baseline.

Deps: Python 3 stdlib only (urllib + threads). No pip install.

Examples
--------
  # Offline sanity check -- parse, synthesize, print the plan, send nothing:
  python replay_session_trace.py trace.jsonl --dry-run

  # A/B against a running frontend (agg launched with --enable-session-radix-cache):
  python replay_session_trace.py trace.jsonl --url http://localhost:8000 --mode on
  python replay_session_trace.py trace.jsonl --url http://localhost:8000 --mode off
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict


def synth_tokens(hashes, block_size, input_length):
    """Mirror mocker synthesize_tokens: token_id = hash as u32, each repeated block_size."""
    toks = []
    for h in hashes:
        tid = h & 0xFFFFFFFF
        toks.extend([tid] * block_size)
        if len(toks) >= input_length:
            return toks[:input_length]
    return toks[:input_length]  # validation guarantees we reach input_length


def load_trace(path):
    """Parse dynamo.agent.trace.v1 JSONL into per-session ordered turn lists."""
    sessions = defaultdict(list)
    skipped = 0
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
        hashes = rp.get("input_sequence_hashes")
        in_len = rp.get("input_length")
        block = rp.get("trace_block_size")
        if not (sid and hashes and in_len and block):
            skipped += 1
            continue
        sessions[sid].append(
            {
                "sid": sid,
                "trajectory_id": ctx.get("trajectory_id"),
                "session_type_id": ctx.get("session_type_id"),
                "model": req.get("model"),
                "recv_ms": req.get("request_received_ms", ev.get("event_time_unix_ms", 0)),
                "end_ms": ev.get("event_time_unix_ms", req.get("request_received_ms", 0)),
                "in_len": in_len,
                "block": block,
                "hashes": hashes,
                "out_tokens": max(1, req.get("output_tokens", 1)),
            }
        )
    for turns in sessions.values():
        turns.sort(key=lambda t: t["recv_ms"])
    return sessions, skipped


def build_body(turn, model, mode, action, timeout, max_tokens_override=None):
    """Construct a /v1/completions body with nvext agent_context + optional session_control."""
    tokens = synth_tokens(turn["hashes"], turn["block"], turn["in_len"])
    body = {
        "model": model or turn["model"],
        "prompt": tokens,
        "max_tokens": max_tokens_override if max_tokens_override is not None else turn["out_tokens"],
        "temperature": 0.0,
        "stream": False,
    }
    nvext = {
        "agent_context": {
            k: v
            for k, v in {
                "session_type_id": turn["session_type_id"] or "replay",
                "session_id": turn["sid"],
                # trajectory_id is required by Dynamo's nvext schema whenever
                # agent_context is present; default to session_id like the provider.
                "trajectory_id": turn["trajectory_id"] or turn["sid"],
                "phase": "reasoning",
            }.items()
            if v is not None
        }
    }
    if mode == "on":
        sc = {"session_id": turn["sid"]}
        if action == "open":
            sc["action"] = "open"
            sc["timeout"] = timeout
        elif action == "close":
            sc["action"] = "close"
        nvext["session_control"] = sc
    body["nvext"] = nvext
    return body


def post(url, body, api_key, timeout_s=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=data,
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
            "x-request-id": str(uuid.uuid4()),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.status, resp.read()


def run_session(turns, args, stats, t0):
    """One thread = one session: stagger to arrival, run turns closed-loop, open->...->close."""
    base = turns[0]["recv_ms"]
    start_offset = (base - args.global_min_ms) / 1000.0 / args.speedup
    sleep_for = t0 + start_offset - time.monotonic()
    if sleep_for > 0:
        time.sleep(sleep_for)

    prev_end = None
    for i, turn in enumerate(turns):
        if prev_end is not None:
            gap = max(0.0, (turn["recv_ms"] - prev_end) / 1000.0 / args.speedup)
            if gap > 0:
                time.sleep(min(gap, args.max_gap_s))
        action = "open" if (i == 0 and args.mode == "on") else None
        body = build_body(turn, args.model, args.mode, action, args.timeout)
        try:
            status, _ = post(args.url, body, args.api_key)
            with stats["lock"]:
                stats["turns"] += 1
                stats["ok" if status == 200 else "err"] += 1
                if action == "open":
                    stats["opens"] += 1
        except (urllib.error.URLError, OSError) as e:
            with stats["lock"]:
                stats["turns"] += 1
                stats["err"] += 1
            print(f"[{turn['sid'][:8]}] turn {i} error: {e}", file=sys.stderr)
        prev_end = turn["end_ms"]

    if args.mode == "on":
        close_body = build_body(turns[-1], args.model, "on", "close", args.timeout, max_tokens_override=1)
        try:
            post(args.url, close_body, args.api_key, timeout_s=10)
            with stats["lock"]:
                stats["closes"] += 1
        except (urllib.error.URLError, OSError):
            pass  # KV cleanup is best-effort, never block on it


def dry_run(sessions, args):
    total_turns = sum(len(v) for v in sessions.values())
    total_in = sum(t["in_len"] for v in sessions.values() for t in v)
    total_out = sum(t["out_tokens"] for v in sessions.values() for t in v)
    span_ms = (
        max(t["recv_ms"] for v in sessions.values() for t in v)
        - min(t["recv_ms"] for v in sessions.values() for t in v)
    )
    print("=== DRY RUN (no requests sent) ===")
    print(f"sessions:        {len(sessions)}")
    print(f"total turns:     {total_turns}")
    print(f"open RPCs:       {len(sessions)} (one per session, first turn)")
    print(f"close RPCs:      {len(sessions)} (one per session, post-last-turn)")
    print(f"input tokens:    {total_in:,}")
    print(f"output tokens:   {total_out:,}")
    print(f"trace span:      {span_ms/1000:.1f}s  (replay wall-clock ~{span_ms/1000/args.speedup:.1f}s at speedup={args.speedup})")
    multi = sum(1 for v in sessions.values() if len(v) > 1)
    print(f"multi-turn sessions: {multi}/{len(sessions)} (these exercise intra-session prefix reuse)")
    # show one sample body, prompt truncated
    sample_sid = max(sessions, key=lambda s: len(sessions[s]))
    turn = sessions[sample_sid][0]
    body = build_body(turn, args.model, args.mode, "open" if args.mode == "on" else None, args.timeout)
    shown = dict(body)
    shown["prompt"] = body["prompt"][:8] + ["...(%d tokens)" % len(body["prompt"])]
    print(f"\nsample first-turn body for session {sample_sid[:8]} ({len(sessions[sample_sid])} turns):")
    print(json.dumps(shown, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="dynamo.agent.trace.v1 JSONL path")
    ap.add_argument("--url", default="http://localhost:8000", help="Dynamo frontend base URL")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default=None, help="override model id (default: from trace)")
    ap.add_argument("--mode", choices=["on", "off"], default="on", help="on=emit session_control; off=baseline")
    ap.add_argument("--speedup", type=float, default=1.0, help="divide wall-clock gaps by this (1.0=real time)")
    ap.add_argument("--max-gap-s", type=float, default=30.0, help="cap any single inter-turn sleep")
    ap.add_argument("--timeout", type=int, default=300, help="session_control open timeout (s)")
    ap.add_argument("--max-sessions", type=int, default=0, help="replay only first N sessions (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="parse + build + print plan, send nothing")
    args = ap.parse_args()

    sessions, skipped = load_trace(args.trace)
    if not sessions:
        print("no replayable request_end events found", file=sys.stderr)
        sys.exit(1)
    if skipped:
        print(f"note: skipped {skipped} non-replayable events", file=sys.stderr)
    if args.max_sessions:
        sessions = dict(list(sessions.items())[: args.max_sessions])
    args.global_min_ms = min(t["recv_ms"] for v in sessions.values() for t in v)

    if args.dry_run:
        dry_run(sessions, args)
        return

    print(f"replaying {len(sessions)} sessions, mode={args.mode}, speedup={args.speedup}x -> {args.url}")
    stats = {"lock": threading.Lock(), "turns": 0, "ok": 0, "err": 0, "opens": 0, "closes": 0}
    t0 = time.monotonic()
    threads = [threading.Thread(target=run_session, args=(turns, args, stats, t0)) for turns in sessions.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    print(
        f"done in {elapsed:.1f}s: {stats['turns']} turns "
        f"({stats['ok']} ok, {stats['err']} err), "
        f"{stats['opens']} opens, {stats['closes']} closes"
    )


if __name__ == "__main__":
    main()
