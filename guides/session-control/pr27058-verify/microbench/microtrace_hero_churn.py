#!/usr/bin/env python3
"""
Micro-trace: one long-lived "hero" agent session vs. a stream of short-lived
"churn" sessions. The smallest workload that makes the close_session benefit
visible per-request rather than as pool aggregates.

  hero   : big context, one turn every --hero-interval seconds (think time),
           alive for the whole run, NEVER closed (it hasn't ended).
  churn  : sessions arrive continuously, do a few quick turns, then end.
           --mode on  -> each churn session is tagged (session_id) and closed
           --mode off -> identical traffic, no session lifecycle

With a small KV pool (--max-total-tokens on the server), OFF-mode churn residue
fills the pool and LRU evicts the hero's idle prefix between its turns -> the
hero's next turn pays a full re-prefill (cached_tokens collapses, latency
spikes). ON-mode churn KV is freed at close, so the hero stays ~fully cached
with flat latency. Compare the per-turn CSVs of two runs.

Requires the server to run with --enable-session-radix-cache,
--radix-eviction-policy=priority and --enable-cache-report (for cached_tokens
in chat usage). Talks /v1/chat/completions like replay_sglang_chat.py.

  python3 microtrace_hero_churn.py --url http://127.0.0.1:8000 --mode on  --out hero_on.csv
  # flush_cache, then:
  python3 microtrace_hero_churn.py --url http://127.0.0.1:8000 --mode off --out hero_off.csv

Python stdlib only. All content is deterministic hex words (~8 tokens/word on
Qwen3); vary --salt to make a rerun's traffic disjoint from resident KV.
"""
import argparse, csv, json, statistics, sys, threading, time, urllib.error, urllib.request, zlib

TOKENS_PER_WORD = 8  # measured for 8-hex-char words on the Qwen3 tokenizer


def words(n_tokens, stream, salt, offset=0):
    """Deterministic hex words worth ~n_tokens, unique per (stream, salt)."""
    n = max(1, round(n_tokens / TOKENS_PER_WORD))
    return " ".join(
        f"{zlib.crc32(f'{salt}:{stream}:{offset + i}'.encode()) & 0xFFFFFFFF:08x}"
        for i in range(n)
    )


def post(url, path, body, timeout_s=300):
    req = urllib.request.Request(
        f"{url}{path}", data=json.dumps(body).encode(), method="POST",
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.status, json.loads(r.read() or b"{}")


def chat(args, content, out_tokens, sid=None):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": out_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    if sid is not None:
        body["session_id"] = sid
    t0 = time.monotonic()
    status, resp = post(args.url, "/v1/chat/completions", body)
    latency = time.monotonic() - t0
    usage = resp.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return status, latency, usage.get("prompt_tokens", 0), details.get("cached_tokens", 0)


def hero_loop(args, rows, stop):
    """One turn every --hero-interval s; context grows a little each turn."""
    base = words(args.hero_context, "hero", args.salt)
    sid = f"hero{args.sid_suffix}" if args.mode == "on" else None
    turn = 0
    t_start = time.monotonic()
    while not stop.is_set():
        turn += 1
        base += " " + words(args.hero_turn_growth, "hero-turn", args.salt, offset=turn * 1000)
        try:
            status, latency, prompt, cached = chat(args, base, args.hero_out_tokens, sid)
            hit = cached / prompt * 100 if prompt else 0.0
            rows.append({"t": round(time.monotonic() - t_start, 1), "turn": turn,
                         "prompt_tokens": prompt, "cached_tokens": cached,
                         "hit_pct": round(hit, 1), "latency_s": round(latency, 3)})
            print(f"[hero] turn {turn:2d}  prompt={prompt:6d}  cached={cached:6d} "
                  f"({hit:5.1f}%)  latency={latency:6.3f}s", flush=True)
        except (urllib.error.URLError, OSError) as e:
            print(f"[hero] turn {turn} error: {e}", file=sys.stderr, flush=True)
        stop.wait(args.hero_interval)


def churn_session(args, n, stats):
    """A short-lived agent: context, a few quick turns, then it ends."""
    sid = f"churn-{n}{args.sid_suffix}" if args.mode == "on" else None
    ctx = words(args.churn_context, f"churn-{n}", args.salt)
    try:
        for turn in range(args.churn_turns):
            ctx += " " + words(150, f"churn-{n}-t", args.salt, offset=turn * 100)
            chat(args, ctx, args.churn_out_tokens, sid)
            time.sleep(1.0)
        if args.mode == "on":
            post(args.url, "/close_session", {"session_id": sid}, timeout_s=10)
            with stats["lock"]:
                stats["closes"] += 1
        with stats["lock"]:
            stats["done"] += 1
    except (urllib.error.URLError, OSError) as e:
        print(f"[churn-{n}] error: {e}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=["on", "off"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--duration", type=float, default=300, help="run length in seconds")
    ap.add_argument("--hero-context", type=int, default=8000, help="hero initial context tokens")
    ap.add_argument("--hero-turn-growth", type=int, default=200, help="tokens added per hero turn")
    ap.add_argument("--hero-interval", type=float, default=25, help="hero think time between turns (s)")
    ap.add_argument("--hero-out-tokens", type=int, default=64)
    ap.add_argument("--churn-context", type=int, default=3000, help="context tokens per churn session")
    ap.add_argument("--churn-turns", type=int, default=3)
    ap.add_argument("--churn-out-tokens", type=int, default=64)
    ap.add_argument("--churn-rate", type=float, default=8, help="churn sessions started per minute")
    ap.add_argument("--salt", default="", help="vary per rerun for disjoint traffic")
    ap.add_argument("--sid-suffix", default="", help="fresh session ids per rerun (closed ids are tombstoned)")
    ap.add_argument("--out", default="hero_turns.csv", help="per-hero-turn CSV output")
    args = ap.parse_args()

    print(f"[micro] mode={args.mode} duration={args.duration}s hero={args.hero_context}tok/"
          f"{args.hero_interval}s churn={args.churn_context}tok x{args.churn_turns} "
          f"@{args.churn_rate}/min -> {args.url}", flush=True)

    rows, stats = [], {"lock": threading.Lock(), "done": 0, "closes": 0}
    stop = threading.Event()
    hero = threading.Thread(target=hero_loop, args=(args, rows, stop))
    hero.start()

    churners, n, t0 = [], 0, time.monotonic()
    while time.monotonic() - t0 < args.duration:
        n += 1
        th = threading.Thread(target=churn_session, args=(args, n, stats))
        th.start()
        churners.append(th)
        time.sleep(60.0 / args.churn_rate)
    stop.set()
    hero.join()
    for th in churners:
        th.join()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "turn", "prompt_tokens", "cached_tokens", "hit_pct", "latency_s"])
        w.writeheader()
        w.writerows(rows)

    lat = [r["latency_s"] for r in rows]
    warm = rows[1:]  # turn 1 is always a cold prefill
    hits = [r["hit_pct"] for r in warm]
    print(f"[micro] done: {len(rows)} hero turns, {stats['done']} churn sessions "
          f"({stats['closes']} closed) -> {args.out}", flush=True)
    if warm:
        print(f"[micro] hero (excl. cold turn 1): hit rate mean={statistics.mean(hits):.1f}% "
              f"min={min(hits):.1f}%  latency median={statistics.median(lat):.3f}s "
              f"p95={sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]:.3f}s max={max(lat):.3f}s", flush=True)


if __name__ == "__main__":
    main()
