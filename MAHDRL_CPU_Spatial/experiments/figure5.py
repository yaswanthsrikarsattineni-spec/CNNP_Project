"""Reproduce paper Figure 5: WiFi interference and reliability vs # WiFi APs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_figure5(csv_path: str = "results/logs/figure5_sweep.csv") -> str:
    df = pd.read_csv(csv_path)
    out = Path("results/graphs/figure5.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))

    # ----- (a) WiFi received interference -----
    ax = axes[0]
    for algo, color, marker, ls in [
        ("MAHDRL", "#1f77b4", "o", "-"),
        ("MADDPG", "#2ca02c", "^", "-."),
    ]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.plot(sub["num_aps"], sub["mean_wifi_dbm"], label=algo,
                color=color, marker=marker, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Number of WiFi APs $N$")
    ax.set_ylabel("Average interference detected by WiFi (dBm)")
    ax.set_title("(a) WiFi received interference")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----- (b) Metaverse user reliability -----
    ax = axes[1]
    for algo, color, marker, ls in [
        ("MAHDRL", "#1f77b4", "o", "-"),
        ("MADDPG", "#2ca02c", "^", "-."),
    ]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.plot(sub["num_aps"], sub["reliability"], label=algo,
                color=color, marker=marker, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Number of WiFi APs $N$")
    ax.set_ylabel("Reliability of Metaverse users")
    ax.set_title("(b) Metaverse user reliability")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure5] Saved: {out}")
    return str(out)


if __name__ == "__main__":
    plot_figure5()