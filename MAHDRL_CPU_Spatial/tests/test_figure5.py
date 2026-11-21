"""Tests for Figure 5 plotting."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def test_plot_figure5_from_synthetic_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "sweep5.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "num_aps", "mean_wifi_dbm", "reliability"])
        for n_ap in [2, 6, 10, 14, 20]:
            w.writerow(["MAHDRL", n_ap, -60.0 - 0.2 * n_ap, 1.0 - 0.005 * n_ap])
            w.writerow(["MADDPG", n_ap, -55.0 - 0.2 * n_ap, 0.98 - 0.008 * n_ap])

    monkeypatch.chdir(tmp_path)
    Path("results/graphs").mkdir(parents=True)

    from experiments import figure5
    import importlib
    importlib.reload(figure5)
    out = figure5.plot_figure5(str(csv_path))

    assert Path(out).exists()