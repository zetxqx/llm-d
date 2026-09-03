"""Charts + stats for the close-session A/B experiment report (2026-09-02 run).

Reads /tmp/close-session-bench/ab-{close,noclose}/reports/per_request_lifecycle_metrics.json
and renders PNGs into the report directory. Prints a stats JSON to stdout.
Style: seaborn whitegrid, despine, dimgrey text, ColorBrewer Dark2.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": "dimgrey",
    "axes.labelcolor": "dimgrey", "axes.titlecolor": "0.2",
    "xtick.color": "dimgrey", "ytick.color": "dimgrey",
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})
DARK2 = sns.color_palette("Dark2")
COL = {"close": DARK2[0], "noclose": DARK2[1]}  # teal vs orange
POOL = 2_340_338  # KV tokens per replica (measured)

OUT = Path("/Users/bobzetian/projects/sessioncontrol/llm-d/guides/session-control/benchmark/close-session-ab-20260902")
OUT.mkdir(parents=True, exist_ok=True)

arms = {}
for arm in ("close", "noclose"):
    rows = []
    data = json.load(open(f"/tmp/close-session-bench/ab-{arm}/reports/per_request_lifecycle_metrics.json"))
    t0 = min(e["start_time"] for e in data)
    for e in data:
        if e.get("error"):
            continue
        info = e["info"]
        rm = info["response_metrics"]
        su = rm.get("server_usage") or {}
        det = su.get("prompt_tokens_details") or {}
        cm = e.get("computed_metrics") or {}
        ttft = cm.get("time_to_first_token")
        if ttft is None and rm.get("chunk_times"):
            ttft = rm["chunk_times"][0] - e["start_time"]
        rows.append(dict(
            t=(e["start_time"] - t0) / 60.0,  # minutes since arm start
            sub="_sa_" in info["graph_event_id"],
            gid=info["graph_event_id"],
            ttft=ttft,
            prompt=su.get("prompt_tokens"),
            cached=det.get("cached_tokens"),
        ))
    rows.sort(key=lambda r: r["t"])
    arms[arm] = rows


def hit(r):
    return (r["cached"] / r["prompt"]) if (r["prompt"] and r["cached"] is not None) else None


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p / 100))] if v else float("nan")


# pool-fill estimate: cumulative newly-computed prefill tokens (prompt - cached)
fill_min = {}
for arm, rows in arms.items():
    cum, t_fill = 0, None
    for r in rows:
        if r["prompt"] is not None and r["cached"] is not None:
            cum += max(0, r["prompt"] - r["cached"])
            if cum >= POOL and t_fill is None:
                t_fill = r["t"]
    fill_min[arm] = t_fill

# ---- stats ----------------------------------------------------------------
stats = {}
for arm, rows in arms.items():
    par = [r for r in rows if not r["sub"]]
    sub = [r for r in rows if r["sub"]]
    fl = fill_min[arm] or 10.0
    late = [r for r in par if r["t"] >= fl]
    early = [r for r in par if r["t"] < fl]

    def block(rs):
        tt = [r["ttft"] for r in rs if r["ttft"] is not None]
        hh = [hit(r) for r in rs if hit(r) is not None]
        # prefix-loss events: non-first turns whose hit ratio collapsed
        losses = [r for r in rs if hit(r) is not None and hit(r) < 0.5
                  and not r["gid"].endswith("turn_0")]
        return dict(n=len(rs), ttft_p50=pct(tt, 50), ttft_p90=pct(tt, 90), ttft_p99=pct(tt, 99),
                    hit_mean=statistics.mean(hh) if hh else None, losses=len(losses))

    stats[arm] = dict(parent=block(par), sub=block(sub), late=block(late), early=block(early),
                      fill_min=fl, total=len(rows))

print(json.dumps(stats, indent=1, default=float))

# ---- fig 1: TTFT ECDF (parent | subagent) ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.subplots_adjust(wspace=0.22, left=0.07, right=0.98, top=0.82, bottom=0.16)
for ax, role, title in ((axes[0], False, "Main-agent turns"), (axes[1], True, "Subagent requests")):
    for arm in ("close", "noclose"):
        tt = sorted(r["ttft"] for r in arms[arm] if r["sub"] == role and r["ttft"] is not None)
        y = np.arange(1, len(tt) + 1) / len(tt) * 100
        ax.step(tt, y, where="post", color=COL[arm], lw=2, label=arm)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 30)
    ax.set_ylim(0, 100)
    ax.set_title(title, loc="left")
    ax.set_xlabel("TTFT seconds (log scale)")
    sns.despine(ax=ax, left=True, bottom=True)
axes[0].set_ylabel("% of requests at or below")
axes[0].legend(loc="lower right")
fig.suptitle("TTFT distribution: close vs noclose (idle cap 1s, 12 sessions/pod, 40 min)",
             x=0.07, ha="left", fontsize=13, fontweight="bold", color="0.2")
fig.savefig(OUT / "ttft-ecdf.png", dpi=180)
plt.close(fig)


def rolling(rows, key, w=51):
    xs = [r["t"] for r in rows]
    ys = [key(r) for r in rows]
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    rx, ry = [], []
    for i in range(len(pts)):
        lo, hi = max(0, i - w // 2), min(len(pts), i + w // 2 + 1)
        rx.append(pts[i][0])
        ry.append(statistics.median(p[1] for p in pts[lo:hi]))
    return rx, ry


# ---- fig 2: rolling TTFT + rolling hit over elapsed time -------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.09, hspace=0.15)
for arm in ("close", "noclose"):
    par = [r for r in arms[arm] if not r["sub"]]
    x, y = rolling(par, lambda r: r["ttft"])
    ax1.plot(x, y, color=COL[arm], lw=1.8, label=arm)
    x, y = rolling(par, hit)
    ax2.plot(x, y, color=COL[arm], lw=1.8)
    if fill_min[arm]:
        for ax in (ax1, ax2):
            ax.axvline(fill_min[arm], color=COL[arm], lw=1, ls=(0, (4, 3)), alpha=0.7)
ax1.set_ylabel("rolling median TTFT (s)")
ax1.legend(loc="upper left")
ax1.set_title("Main-agent turns over the run; dashed lines = estimated KV-pool fill (eviction onset)", loc="left")
ax2.set_ylabel("rolling median cache-hit ratio")
ax2.set_xlabel("minutes since arm start")
ax2.set_ylim(0, 1.02)
for ax in (ax1, ax2):
    sns.despine(ax=ax, left=True, bottom=True)
fig.suptitle("Timeline: the arms stay superimposed before AND after eviction begins",
             x=0.08, ha="left", fontsize=13, fontweight="bold", color="0.2")
fig.savefig(OUT / "timeline.png", dpi=180)
plt.close(fig)

# ---- fig 3: cumulative new-KV written vs pool ------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.14)
for arm in ("close", "noclose"):
    xs, ys, cum = [], [], 0
    for r in arms[arm]:
        if r["prompt"] is not None and r["cached"] is not None:
            cum += max(0, r["prompt"] - r["cached"])
            xs.append(r["t"])
            ys.append(cum / 1e6)
    ax.plot(xs, ys, color=COL[arm], lw=2, label=arm)
ax.axhline(POOL / 1e6, color="dimgrey", lw=1.2, ls=(0, (4, 3)))
ax.annotate("KV pool per pod (2.34M tokens)", xy=(1, POOL / 1e6), xytext=(1, POOL / 1e6 + 0.15),
            color="dimgrey", fontsize=10)
ax.set_title("Cumulative newly-computed KV tokens: both arms overwrite the pool ~4.3x (heavy eviction)", loc="left")
ax.set_xlabel("minutes since arm start")
ax.set_ylabel("million tokens")
ax.legend(loc="upper left")
sns.despine(ax=ax, left=True, bottom=True)
fig.savefig(OUT / "kv-writes.png", dpi=180)
plt.close(fig)

# ---- fig 4: prefix-loss events per 5-min bucket ----------------------------
fig, ax = plt.subplots(figsize=(9, 4.0))
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14)
buckets = np.arange(0, 45, 5)
width = 1.8
for k, arm in enumerate(("close", "noclose")):
    par = [r for r in arms[arm] if not r["sub"]]
    losses = [r["t"] for r in par if hit(r) is not None and hit(r) < 0.5
              and not r["gid"].endswith("turn_0")]
    counts, _ = np.histogram(losses, bins=list(buckets) + [45])
    ax.bar(buckets + 0.6 + k * width, counts, width=width, align="edge",
           color=COL[arm], label=arm, edgecolor="white", linewidth=0.5)
ax.set_title("Prefix-loss events (non-first turns with cache-hit < 50%) per 5-minute bucket", loc="left")
ax.set_xlabel("minutes since arm start")
ax.set_ylabel("events")
ax.set_xticks(buckets)
ax.legend()
sns.despine(ax=ax, left=True, bottom=True)
fig.savefig(OUT / "prefix-loss.png", dpi=180)
plt.close(fig)

print("figures written to", OUT)
