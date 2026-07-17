#!/usr/bin/env python3
"""
Replay a captured agent trace against a SINGLE SGLang worker through the
OpenAI-compatible chat API -- the same A/B as replay_sglang_native.py, but via
/v1/chat/completions instead of the native /generate:

  - tag a request: POST /v1/chat/completions with the SGLang extension field
                   "session_id" (an OpenAI SDK passes it via extra_body)
  - release:       POST /close_session {"session_id": <session_id>}

Because chat/completions takes text (the server tokenizes and applies the chat
template), the trace's synthetic token hashes are rendered as deterministic hex
words instead of raw input_ids: same hash prefix -> same text prefix, so the
prefix-sharing structure of the trace is preserved. Token counts are therefore
approximate (~block_size tokens per hash block via --tokens-per-word); read the
exact numbers back from the response `usage`, which this script aggregates.

  python replay_sglang_chat.py golden-trace-12x4.jsonl --mode on  --speedup 4
  python replay_sglang_chat.py golden-trace-12x4.jsonl --mode off --speedup 4

--mode on  -> session_id tag + /close_session per session (the feature)
--mode off -> identical workload, no session marking (plain radix baseline)
--salt     -> mixes into every hex word; use a different salt per wave the way
              the native script varies --vocab (same salt = 100% cache hit on
              a warm pod, different salt = disjoint new traffic)
--sid-suffix -> fresh session ids for reruns; closed ids are tombstoned
              server-side and silently never re-tag KV

Watch the same metrics as the native replay: sglang:kv_evictable_tokens and
sglang:evicted_tokens_total. Python stdlib only.
"""
import argparse, json, sys, threading, time, urllib.error, urllib.request, zlib
from collections import defaultdict


def synth_text(hashes, block_size, input_length, tokens_per_word, salt):
    """Render hash blocks as deterministic hex words, ~block_size tokens per block."""
    salt_mix = zlib.crc32(salt.encode()) if salt else 0
    words_per_block = max(1, round(block_size / tokens_per_word))
    n_blocks = max(1, round(input_length / block_size))
    words = []
    for h in hashes[:n_blocks]:
        word = f"{(h ^ salt_mix) & 0xFFFFFFFF:08x}"
        words.extend([word] * words_per_block)
    return " ".join(words)


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


def post(url, path, body, timeout_s=300):
    req = urllib.request.Request(
        f"{url}{path}", data=json.dumps(body).encode(), method="POST",
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.status, json.loads(r.read() or b"{}")


def chat_body(turn, args):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": synth_text(
            turn["hashes"], turn["block"], turn["in_len"], args.tokens_per_word, args.salt)}],
        "max_tokens": turn["out_tokens"],
        "temperature": 0.0,
        # SGLang extension: fixed-length outputs so gibberish prompts still
        # reproduce the trace's output token counts.
        "ignore_eos": True,
        "stream": False,
    }
    if args.mode == "on":
        # SGLang extension field (extra_body for an OpenAI SDK): radix-native
        # session tag, implicit open, released by /close_session. Do NOT use
        # session_params -- that selects the old dedicated-slot controller.
        body["session_id"] = turn["sid"] + args.sid_suffix
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
            status, resp = post(args.url, "/v1/chat/completions", chat_body(turn, args))
            usage = resp.get("usage") or {}
            details = usage.get("prompt_tokens_details")
            with stats["lock"]:
                stats["turns"] += 1
                stats["ok" if status == 200 else "err"] += 1
                stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                stats["completion_tokens"] += usage.get("completion_tokens", 0)
                if details is not None:
                    stats["details_seen"] = True
                    stats["cached_tokens"] += details.get("cached_tokens", 0)
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
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="SGLang server base URL")
    ap.add_argument("--mode", choices=["on", "off"], default="on")
    ap.add_argument("--model", default="Qwen/Qwen3-32B", help="model name for the OpenAI endpoint")
    ap.add_argument("--speedup", type=float, default=4.0)
    ap.add_argument("--max-gap-s", type=float, default=2.0)
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--tokens-per-word", type=int, default=4,
                    help="approx tokens one hex word costs; sizes words-per-block")
    ap.add_argument("--salt", default="", help="mixes into every hex word; vary per wave for disjoint traffic")
    ap.add_argument("--sid-suffix", default="", help="append to session ids; needed for reruns because closed session ids are tombstoned server-side and can never re-tag KV")
    args = ap.parse_args()

    sessions = load_trace(args.trace)
    if not sessions:
        print("no replayable request_end events", file=sys.stderr); sys.exit(1)
    if args.max_sessions:
        sessions = dict(list(sessions.items())[: args.max_sessions])
    args.global_min_ms = min(t["recv_ms"] for v in sessions.values() for t in v)

    print(f"[sglang-chat] replaying {len(sessions)} sessions, mode={args.mode}, "
          f"speedup={args.speedup}x, salt={args.salt!r} -> {args.url}/v1/chat/completions")
    stats = {"lock": threading.Lock(), "turns": 0, "ok": 0, "err": 0, "closes": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "details_seen": False}
    t0 = time.monotonic()
    ts = [threading.Thread(target=run_session, args=(v, args, stats, t0)) for v in sessions.values()]
    for t in ts: t.start()
    for t in ts: t.join()
    print(f"[sglang-chat] done in {time.monotonic()-t0:.1f}s: {stats['turns']} turns "
          f"({stats['ok']} ok, {stats['err']} err), {stats['closes']} /close_session")
    if stats["details_seen"]:
        hit = stats["cached_tokens"] / stats["prompt_tokens"] * 100 if stats["prompt_tokens"] else 0.0
        cached = f"{stats['cached_tokens']:,} cached ({hit:.1f}% prefix hit)"
    else:
        # sglang's chat responses return prompt_tokens_details: null — cache
        # hits are invisible here; watch sglang:kv_evictable_tokens instead.
        cached = "cached n/a (server omits prompt_tokens_details)"
    print(f"[sglang-chat] usage: {stats['prompt_tokens']:,} prompt / {stats['completion_tokens']:,} completion "
          f"tokens, {cached}")


if __name__ == "__main__":
    main()
