"""Reproduce paper Figure 3 — reward and data-rate convergence."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    ("MAHDRL",                        "results/logs/mahdrl_training.csv",           "#1f77b4", "o", "-"),
    ("MAHDRL without RIS",            "results/logs/mahdrl_no_ris_training.csv",    "#ff7f0e", "s", "--"),
    ("MADDPG",                        "results/logs/maddpg_training.csv",           "#2ca02c", "^", "-."),
    ("CC-MAHDRL (M=1 degenerate)",    "results/logs/cc_mahdrl_training.csv",        "#d62728", "d", ":"),
]

SMOOTH_WINDOW = 10


def _rolling(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    padded = np.convolve(x, kernel, mode="same")
    padded[:window] = x[:window]
    padded[-window:] = x[-window:]
    return padded


def main():
    loaded = []
    for label, csv_path, color, marker, ls in RUNS:
        p = Path(csv_path)
        if not p.exists():
            print(f"[figure3] SKIP (not found): {csv_path}")
            continue
        df = pd.read_csv(p)
        print(f"[figure3] Loaded {csv_path}: {len(df)} episodes")
        loaded.append((label, df, color, marker, ls))

    if not loaded:
        raise RuntimeError("No training CSVs found. Run training first.")

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    ax = axes[0]
    for label, df, color, marker, ls in loaded:
        y = _rolling(df["episode_reward_bps"].to_numpy() / 1e6, SMOOTH_WINDOW)
        ax.plot(df["episode"], y, label=label, color=color, marker=marker,
                linestyle=ls, markersize=4, markevery=20, linewidth=1.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rewards (Mbps-eq)")
    ax.set_title("(a) Convergence of the rewards")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    for label, df, color, marker, ls in loaded:
        y = _rolling(df["episode_rate_bps"].to_numpy() / 1e6, SMOOTH_WINDOW)
        ax.plot(df["episode"], y, label=label, color=color, marker=marker,
                linestyle=ls, markersize=4, markevery=20, linewidth=1.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average data rate of each BS (Mbps)")
    ax.set_title("(b) Convergence of each BS's average data rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()

    out = Path("results/graphs/figure3.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[figure3] Saved: {out}")
    for label, df, *_ in loaded:
        r = df["episode_reward_bps"].to_numpy() / 1e6
        rr = df["episode_rate_bps"].to_numpy() / 1e6
        print(f"  {label:40s}  reward={r[0]:5.2f}→{r[-1]:5.2f}   "
              f"rate={rr[0]:5.2f}→{rr[-1]:5.2f} Mbps")


if __name__ == "__main__":
    main()