"""Reproduce paper Figure 4: rate/latency/Wi-Fi vs. number of Metaverse users."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_figure4(csv_path: str = "results/logs/figure4_sweep.csv") -> str:
    df = pd.read_csv(csv_path)
    out = Path("results/graphs/figure4.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 11))

    # ----- (a) Average data rate per user -----
    ax = axes[0]
    for algo, color, marker, ls in [
        ("MAHDRL", "#1f77b4", "o", "-"),
        ("MADDPG", "#2ca02c", "^", "-."),
    ]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["per_user_rate_mbps"], label=algo,
                color=color, marker=marker, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Number of Metaverse users $K_m$")
    ax.set_ylabel("Average data rate of each user (Mbps)")
    ax.set_title("(a) Average data rate per Metaverse user")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----- (b) Average latency -----
    ax = axes[1]
    for algo, color, marker, ls in [
        ("MAHDRL", "#1f77b4", "o", "-"),
        ("MADDPG", "#2ca02c", "^", "-."),
    ]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["mean_latency_ms"], label=algo,
                color=color, marker=marker, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Number of Metaverse users $K_m$")
    ax.set_ylabel("Average latency of each user (ms)")
    ax.set_title("(b) Average latency per Metaverse user")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----- (c) Wi-Fi interference -----
    ax = axes[2]
    for algo, color, marker, ls in [
        ("MAHDRL", "#1f77b4", "o", "-"),
        ("MADDPG", "#2ca02c", "^", "-."),
    ]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["mean_wifi_dbm"], label=algo,
                color=color, marker=marker, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Number of Metaverse users $K_m$")
    ax.set_ylabel("Average interference detected by Wi-Fi (dBm)")
    ax.set_title("(c) Wi-Fi received interference")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure4] Saved: {out}")
    return str(out)


if __name__ == "__main__":
    plot_figure4()