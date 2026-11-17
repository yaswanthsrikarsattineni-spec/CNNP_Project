"""Reproduce paper Figure 6: rate and WiFi interference vs # RIS elements."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_figure6(csv_path: str = "results/logs/figure6_sweep.csv") -> str:
    df = pd.read_csv(csv_path)
    out = Path("results/graphs/figure6.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))

    # ----- (a) Average data rate -----
    ax = axes[0]
    for algo, color, marker, ls in [
        ("MAHDRL", "#1f77b4", "o", "-"),
        ("MADDPG", "#2ca02c", "^", "-."),
    ]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.plot(sub["num_elements"], sub["total_rate_mbps"], label=algo,
                color=color, marker=marker, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Number of RIS elements $L$")
    ax.set_ylabel("Average data rate of each BS (Mbps)")
    ax.set_title("(a) Average data rate of each BS")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----- (b) Wi-Fi interference -----
    ax = axes[1]
    for algo, color, marker, ls in [
        ("MAHDRL", "#1f77b4", "o", "-"),
        ("MADDPG", "#2ca02c", "^", "-."),
    ]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.plot(sub["num_elements"], sub["mean_wifi_dbm"], label=algo,
                color=color, marker=marker, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Number of RIS elements $L$")
    ax.set_ylabel("Average interference detected by WiFi (dBm)")
    ax.set_title("(b) WiFi received interference")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure6] Saved: {out}")
    return str(out)


if __name__ == "__main__":
    plot_figure6()