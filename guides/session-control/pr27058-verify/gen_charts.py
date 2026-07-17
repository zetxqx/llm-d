# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
# ]
# ///
"""Render the PR 27058 A/B KV-pressure figure from the sampled metrics CSVs.

Styled with the matplotlib skill (whitegrid, despined, DIVERGING_COLORBLIND
comparison colors). Layout: 4x2 grid — columns are pass 1 / pass 2, rows are
the four sampled sglang gauges: resident cached KV (kv_evictable_tokens),
cumulative forced evictions (evicted_tokens_total), active KV usage
(token_usage x pool, the in-flight load envelope), and free pool headroom
(kv_available_tokens). Each panel names its source metric in the right-hand
title. ON arm = replay with
close_session, OFF arm = no session lifecycle; arms ran on separate pods, so
each series is aligned to its own replay start (first sample under load).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter

POOL = 321_614
BLUE = "#4575b4"  # ON  (with close_session)
RED = "#d73027"   # OFF (no close_session)


def load(path):
    """Load one sampler CSV and align time to the arm's replay start.

    Parameters
    ----------
    path : str
        Sampler CSV with columns t, token_usage, kv_available, kv_evictable, evicted_total.

    Returns
    -------
    pandas.DataFrame
        Columns t (seconds since replay start), token_usage (pool fraction),
        kv_available, kv_evictable, evicted_total.
    """
    df = pd.read_csv(path)
    df = df[df["token_usage"] != "NA"].astype(float).dropna()
    start = df.loc[df["token_usage"] > 0.01, "t"].min()
    df["t"] = df["t"] - start
    return df[df["t"] >= -5]


def main():
    """Render the 3x2 A/B figure.

    Saves
    -----
    ./figures/kv-pressure-ab.png at 150 DPI.
    """
    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    kfmt = FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0")

    # --- Data ---
    passes = {
        "Pass 1 · flushed caches": (load("metrics_on.csv"), load("metrics_off.csv")),
        "Pass 2 · second wave, no flush": (load("metrics_on3.csv"), load("metrics_off2.csv")),
    }
    evict_max = max(df["evicted_total"].max() for pair in passes.values() for df in pair)
    usage_max = max(df["token_usage"].max() for pair in passes.values() for df in pair) * POOL

    # --- Plot ---
    fig = plt.figure(figsize=(11, 15), dpi=150)
    gs = gridspec.GridSpec(4, 2)
    gs.update(wspace=0.08, hspace=0.32, left=0.08, right=0.99, top=0.915, bottom=0.06)
    fig.suptitle("Session radix cache A/B — sglang v0.5.15.post1, Qwen3-32B TP2, 69-session trace at 4x",
                 fontsize=14, y=0.985, color="dimgrey")

    labels = [chr(ord("a") + i) for i in range(8)]
    for col, (pass_title, (on, off)) in enumerate(passes.items()):
        ax_kv = plt.subplot(gs[0, col])
        ax_ev = plt.subplot(gs[1, col])
        ax_tu = plt.subplot(gs[2, col])
        ax_av = plt.subplot(gs[3, col])

        # Pass description lives in one column header; panel titles stay short
        # so the right-hand metric labels don't collide
        pos = ax_kv.get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.022, pass_title,
                 ha="center", va="bottom", fontsize=12, weight="semibold", color="dimgrey")

        # Row 1: resident cached KV vs pool capacity — filled to make occupancy tangible
        ax_kv.axhline(POOL, color="lightgrey", linewidth=1.2, linestyle="--", zorder=1)
        for df, color, z in ((off, RED, 3), (on, BLUE, 4)):
            ax_kv.fill_between(df["t"], df["kv_evictable"], color=color, alpha=0.12, zorder=z - 1)
            ax_kv.plot(df["t"], df["kv_evictable"], color=color, linewidth=2, zorder=z)
        # End-state annotations: residue kept vs released
        end_off = off["kv_evictable"].iloc[-1]
        ax_kv.text(off["t"].iloc[-1] - 3, end_off - POOL * 0.10, f"{end_off/1000:.0f}k kept",
                   ha="right", va="top", fontsize=9, weight="semibold", color=RED)
        ax_kv.text(on["t"].iloc[-1] - 3, on["kv_evictable"].iloc[-1] + POOL * 0.03, "released",
                   ha="right", va="bottom", fontsize=9, weight="semibold", color=BLUE)
        ax_kv.set_ylim(0, POOL * 1.14)
        ax_kv.set_title(r"$\bf{(" + labels[col] + r")}$" + "  Resident cached KV",
                        loc="left", fontsize=11, pad=7, color="dimgrey")
        if col == 0:
            ax_kv.text(on["t"].max(), POOL + POOL * 0.025, f"pool capacity {POOL/1000:.0f}k tokens",
                       ha="right", va="bottom", fontsize=8.5, color="dimgrey")
            ax_kv.legend(handles=ax_kv.get_lines()[1:3][::-1],
                         labels=["ON — with close_session", "OFF — no close_session"],
                         loc="center left", fontsize=9, frameon=True, facecolor="white",
                         framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey")

        # Row 2: cumulative forced evictions
        ax_ev.plot(off["t"], off["evicted_total"], color=RED, linewidth=2, zorder=3)
        ax_ev.plot(on["t"], on["evicted_total"], color=BLUE, linewidth=2, zorder=4)
        ax_ev.set_ylim(0, evict_max * 1.14)
        ax_ev.set_title(r"$\bf{(" + labels[col + 2] + r")}$" + "  Forced LRU evictions",
                        loc="left", fontsize=11, pad=7, color="dimgrey")

        ends = {color: df["evicted_total"].iloc[-1] for df, color in ((off, RED), (on, BLUE))}
        if all(v == 0 for v in ends.values()):
            ax_ev.text(0.5, 0.5, "no forced evictions in either arm", transform=ax_ev.transAxes,
                       ha="center", va="center", fontsize=10, color="dimgrey", style="italic")
        else:
            for df, color in ((off, RED), (on, BLUE)):
                end = df["evicted_total"].iloc[-1]
                ax_ev.text(df["t"].iloc[-1], end + evict_max * 0.03, f"{end:,.0f}",
                           ha="right", va="bottom", fontsize=9, weight="semibold", color=color)

        # Row 3: active KV usage (in-flight load) — same load envelope in both
        # arms shows the eviction gap is residue, not traffic
        for df, color, z in ((off, RED, 3), (on, BLUE, 4)):
            ax_tu.plot(df["t"], df["token_usage"] * POOL, color=color, linewidth=2, zorder=z)
        peak = max(off["token_usage"].max(), on["token_usage"].max())
        ax_tu.axhline(peak * POOL, color="lightgrey", linewidth=1.2, linestyle="--", zorder=1)
        ax_tu.text(max(off["t"].max(), on["t"].max()), peak * POOL + usage_max * 0.03,
                   f"both arms peak at {peak:.0%} of pool",
                   ha="right", va="bottom", fontsize=8.5, color="dimgrey")
        ax_tu.set_ylim(0, usage_max * 1.20)
        ax_tu.set_title(r"$\bf{(" + labels[col + 4] + r")}$" + "  Active KV usage",
                        loc="left", fontsize=11, pad=7, color="dimgrey")

        # Row 4: free pool headroom — the complement view: ON hands the pool
        # back, OFF ends the run nearly exhausted
        ax_av.axhline(POOL, color="lightgrey", linewidth=1.2, linestyle="--", zorder=1)
        for df, color, z in ((off, RED, 3), (on, BLUE, 4)):
            ax_av.plot(df["t"], df["kv_available"], color=color, linewidth=2, zorder=z)
        for df, color, dy in ((off, RED, POOL * 0.04), (on, BLUE, -POOL * 0.10)):
            end = df["kv_available"].iloc[-1]
            ax_av.text(df["t"].iloc[-1] - 3, end + dy, f"{end/1000:.0f}k free",
                       ha="right", va="bottom" if dy > 0 else "top",
                       fontsize=9, weight="semibold", color=color)
        ax_av.set_ylim(0, POOL * 1.14)
        ax_av.set_title(r"$\bf{(" + labels[col + 6] + r")}$" + "  Free KV pool",
                        loc="left", fontsize=11, pad=7, color="dimgrey")
        ax_av.set_xlabel("Seconds since replay start", fontsize=9, labelpad=6, color="dimgrey")

        # Right-hand titles name the sampled Prometheus gauge for each row
        metrics = {ax_kv: "sglang:kv_evictable_tokens",
                   ax_ev: "sglang:evicted_tokens_total{RadixCache}",
                   ax_tu: "sglang:token_usage × pool",
                   ax_av: "sglang:kv_available_tokens"}
        for ax, metric in metrics.items():
            ax.set_title(metric, loc="right", fontsize=8, pad=7,
                         color="darkgrey", family="monospace")

        for ax in (ax_kv, ax_ev, ax_tu, ax_av):
            ax.grid(False)
            ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
            ax.yaxis.set_major_formatter(kfmt)
            ax.patch.set_edgecolor("lightgrey")
            ax.patch.set_linewidth(0.8)
            if col == 0:
                ax.set_ylabel("Tokens", fontsize=9, labelpad=6, color="dimgrey")
            else:
                ax.tick_params(labelleft=False)

    fig.text(0.99, 0.015,
             "Without close_session, dead-session residue fills the pool in pass 1 (b top ≈ d start) "
             "and forces 627k tokens of evictions in pass 2; with close_session: zero.",
             ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic")

    sns.despine(left=True, bottom=True)

    # --- Save ---
    Path("./figures").mkdir(exist_ok=True)
    plt.savefig("./figures/kv-pressure-ab.png", dpi=150, bbox_inches="tight")
    print("wrote figures/kv-pressure-ab.png")


if __name__ == "__main__":
    main()
