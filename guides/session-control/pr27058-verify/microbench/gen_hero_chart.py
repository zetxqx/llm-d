# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
# ]
# ///
"""Render the hero-vs-churn micro benchmark figure.

Styled with the matplotlib skill (whitegrid, despined, DIVERGING_COLORBLIND
comparison colors). 2x2 grid: top row is the hero's per-turn experience (cache
hit rate, turn latency), bottom row is the pod-level cause (resident cached KV
vs pool, cumulative forced evictions). ON = churn sessions closed, OFF = same
traffic with no lifecycle.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter

POOL = 40_000
BLUE = "#4575b4"  # ON  (churn closed)
RED = "#d73027"   # OFF (no lifecycle)


def load_pod(path):
    df = pd.read_csv(path)
    df = df[df["token_usage"] != "NA"].astype(float).dropna()
    start = df.loc[df["token_usage"] > 0.01, "t"].min()
    df["t"] = df["t"] - start
    # evicted_total is a lifetime counter (survives flush_cache) — zero to run start
    df["evicted_total"] -= df["evicted_total"].iloc[0]
    return df[df["t"] >= -5]


def main():
    """Render the 2x2 micro benchmark figure.

    Saves
    -----
    ./figures/hero-vs-churn.png at 150 DPI.
    """
    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    kfmt = FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0")

    # --- Data ---
    hero_on = pd.read_csv("hero_on.csv")
    hero_off = pd.read_csv("hero_off.csv")
    pod_on = load_pod("pod_metrics_on.csv")
    pod_off = load_pod("pod_metrics_off.csv")

    # --- Plot ---
    fig = plt.figure(figsize=(11, 8), dpi=150)
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.24, hspace=0.32, left=0.08, right=0.99, top=0.88, bottom=0.11)
    fig.suptitle("One live agent vs. dead-session churn — 40k-token pool, 24 churn sessions/min, 40s think time",
                 fontsize=13, y=0.965, color="dimgrey")

    ax_hit = plt.subplot(gs[0, 0])
    ax_lat = plt.subplot(gs[0, 1])
    ax_kv = plt.subplot(gs[1, 0])
    ax_ev = plt.subplot(gs[1, 1])

    # (a) hero per-turn cache hit rate — the headline
    for df, color, label in ((hero_off, RED, "OFF — churn never closed"),
                             (hero_on, BLUE, "ON — churn closed on end")):
        warm = df[df["turn"] > 1]
        ax_hit.plot(warm["turn"], warm["hit_pct"], color=color, linewidth=2,
                    marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.8,
                    label=label)
    ax_hit.set_ylim(-5, 105)
    ax_hit.set_title(r"$\bf{(a)}$" + "  Hero cache hit rate per turn (%)", loc="left",
                     fontsize=11, pad=7, color="dimgrey")
    ax_hit.set_xlabel("Hero turn (turn 1 = cold start, excluded)", fontsize=9, labelpad=6, color="dimgrey")
    ax_hit.legend(loc="center right", fontsize=9, frameon=True, facecolor="white",
                  framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey")

    # (b) hero per-turn latency
    for df, color in ((hero_off, RED), (hero_on, BLUE)):
        warm = df[df["turn"] > 1]
        ax_lat.plot(warm["turn"], warm["latency_s"], color=color, linewidth=2,
                    marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    med_on = hero_on[hero_on["turn"] > 1]["latency_s"].median()
    med_off = hero_off[hero_off["turn"] > 1]["latency_s"].median()
    ax_lat.text(0.98, 0.96, f"median {med_off:.2f}s vs {med_on:.2f}s  (+{(med_off/med_on-1)*100:.0f}%)",
                transform=ax_lat.transAxes, ha="right", va="top", fontsize=9,
                color="dimgrey", style="italic")
    ax_lat.set_ylim(0, max(hero_off["latency_s"].max(), hero_on["latency_s"].max()) * 1.25)
    ax_lat.set_title(r"$\bf{(b)}$" + "  Hero turn latency (s)", loc="left",
                     fontsize=11, pad=7, color="dimgrey")
    ax_lat.set_xlabel("Hero turn", fontsize=9, labelpad=6, color="dimgrey")

    # (c) pod resident cached KV — why: OFF pool pinned full of residue
    ax_kv.axhline(POOL, color="lightgrey", linewidth=1.2, linestyle="--", zorder=1)
    ax_kv.text(pod_off["t"].max(), POOL + POOL * 0.02, f"pool {POOL/1000:.0f}k",
               ha="right", va="bottom", fontsize=8.5, color="dimgrey")
    for df, color in ((pod_off, RED), (pod_on, BLUE)):
        ax_kv.fill_between(df["t"], df["kv_evictable"], color=color, alpha=0.12)
        ax_kv.plot(df["t"], df["kv_evictable"], color=color, linewidth=2)
    ax_kv.set_ylim(0, POOL * 1.16)
    ax_kv.yaxis.set_major_formatter(kfmt)
    ax_kv.set_title(r"$\bf{(c)}$" + "  Pod resident cached KV (tokens)", loc="left",
                    fontsize=11, pad=7, color="dimgrey")
    ax_kv.set_xlabel("Seconds since run start", fontsize=9, labelpad=6, color="dimgrey")

    # (d) pod cumulative forced evictions
    for df, color in ((pod_off, RED), (pod_on, BLUE)):
        ax_ev.plot(df["t"], df["evicted_total"], color=color, linewidth=2)
        end = df["evicted_total"].iloc[-1]
        ax_ev.text(df["t"].iloc[-1], end + pod_off["evicted_total"].max() * 0.03,
                   f"{end:,.0f}", ha="right", va="bottom", fontsize=9,
                   weight="semibold", color=color)
    ax_ev.set_ylim(0, pod_off["evicted_total"].max() * 1.16)
    ax_ev.yaxis.set_major_formatter(kfmt)
    ax_ev.set_title(r"$\bf{(d)}$" + "  Pod forced LRU evictions (tokens, cumulative)", loc="left",
                    fontsize=11, pad=7, color="dimgrey")
    ax_ev.set_xlabel("Seconds since run start", fontsize=9, labelpad=6, color="dimgrey")

    for ax in (ax_hit, ax_lat, ax_kv, ax_ev):
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)

    fig.text(0.99, 0.015,
             "Same agent, same churn traffic: with closes the hero stays ~98% cached at flat latency; "
             "without, every think pause ends in a full re-prefill.",
             ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic")

    sns.despine(left=True, bottom=True)

    # --- Save ---
    Path("./figures").mkdir(exist_ok=True)
    plt.savefig("./figures/hero-vs-churn.png", dpi=150, bbox_inches="tight")
    print("wrote figures/hero-vs-churn.png")


if __name__ == "__main__":
    main()
