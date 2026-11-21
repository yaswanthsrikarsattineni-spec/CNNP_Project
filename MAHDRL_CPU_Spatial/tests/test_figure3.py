"""Tests for Figure 3 plotting (uses synthetic data)."""
import csv
from pathlib import Path

import pandas as pd


def test_training_csv_can_be_created_and_loaded(tmp_path):
    """Simulate the CSV that train_mahdrl.py writes, load it back."""
    csv_path = tmp_path / "fake.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "total_step", "episode_reward_bps",
            "episode_rate_bps", "episode_wifi_excess_db",
            "episode_latency_ms", "reliability",
        ])
        for ep in range(1, 21):
            w.writerow([ep, ep * 100, 18e6 + ep * 1e4,
                        20e6 + ep * 1e4, 100.0, 5.0, 1.0])
    df = pd.read_csv(csv_path)
    assert len(df) == 20
    assert "episode_reward_bps" in df.columns
    assert "episode_rate_bps" in df.columns


def test_rolling_smoothing_works():
    """Smoothed signal must have lower high-frequency noise than input."""
    import numpy as np
    from experiments.figure3 import _rolling

    rng = np.random.default_rng(0)
    # Noisy upward trend: signal + gaussian noise
    x = np.linspace(0, 10, 200) + rng.normal(0, 1.0, size=200)
    y = _rolling(x, 20)

    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
    # Smoothed signal must have lower variance in its derivative
    assert np.std(np.diff(y[30:-30])) < np.std(np.diff(x[30:-30]))


def test_plot_generates_png(tmp_path, monkeypatch):
    """End-to-end: fake CSV → plot → verify PNG exists."""
    import matplotlib
    matplotlib.use("Agg")

    # Create a fake training CSV inside a sandbox directory
    sandbox = tmp_path
    (sandbox / "results" / "logs").mkdir(parents=True)
    (sandbox / "results" / "graphs").mkdir(parents=True)

    fake_csv = sandbox / "results" / "logs" / "mahdrl_training.csv"
    with open(fake_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "total_step", "episode_reward_bps",
            "episode_rate_bps", "episode_wifi_excess_db",
            "episode_latency_ms", "reliability",
        ])
        for ep in range(1, 21):
            w.writerow([ep, ep * 100, 18e6, 20e6, 100.0, 5.0, 1.0])

    # Change CWD to sandbox for the plot call
    monkeypatch.chdir(sandbox)
    import importlib
    from experiments import figure3 as f3
    importlib.reload(f3)
    f3.main()

    png = sandbox / "results" / "graphs" / "figure3.png"
    assert png.exists(), f"Plot not saved: {png}"