"""Tests for Figure 4 plotting."""
import csv
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")


def test_plot_figure4_from_synthetic_csv(tmp_path, monkeypatch):
    """Feed a synthetic sweep CSV → verify PNG is produced."""
    csv_path = tmp_path / "sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "K", "per_user_rate_mbps",
                    "mean_latency_ms", "mean_wifi_dbm"])
        for K in [6, 8, 10, 12]:
            w.writerow(["MAHDRL", K, 50.0/K, 1.0*K, -100.0 - 0.5*K])
            w.writerow(["MADDPG", K, 40.0/K, 1.5*K, -95.0 - 0.5*K])

    monkeypatch.chdir(tmp_path)
    Path("results/graphs").mkdir(parents=True)

    from experiments import figure4
    import importlib
    importlib.reload(figure4)
    out = figure4.plot_figure4(str(csv_path))

    assert Path(out).exists()