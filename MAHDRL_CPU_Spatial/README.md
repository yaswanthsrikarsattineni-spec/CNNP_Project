# MAHDRL Replication — IEEE JSAC 2024

Python/PyTorch replication of *"Human-Aware Dynamic Hierarchical Network Control for Distributed Metaverse Services"* (IEEE JSAC, Vol. 42, No. 3, March 2024).

## Overview
This repository implements the **MAHDRL** (Multi-Agent Hierarchical Deep Reinforcement Learning) algorithm from scratch, including:
- **Environment**: RIS-assisted NR-U network, Rician channels, Lyapunov queue stability, Wi-Fi coexistence (20+ equations from the paper).
- **Agents**: DQN (discrete user selection), DDPG (power & RIS phases), QMIX (mixing network).
- **Baselines**: MADDPG (continuous relaxation), MAHDRL-no-RIS, CC-MAHDRL.
- **Results**: All 4 main figures (Fig 3, 4, 5, 6) reproduced.

## Project Structure
- `environment/`: Physics engine (channels, SINR, rates, queues, RIS).
- `agents/`: Neural networks (DQN, DDPG, Mixer).
- `training/`: Training loops for MAHDRL, MADDPG, etc.
- `experiments/`: Sweep runners and plotting scripts.
- `tests/`: **110 unit tests** covering all physics and agent modules.

## How to Run
Ensure Python 3.10+ and dependencies (`pip install -r requirements.txt`) are installed.

### 1. Run all tests (verifies everything works)
```bash
pytest -v

# Figure 3: Convergence (Takes ~30 mins)
python run_fig3_experiment.py --algo all --episodes 100

# Figure 4: User sweep (Takes ~40 mins)
python -m experiments.run_figure4

# Figure 5: WiFi AP sweep (Takes ~60 mins)
python -m experiments.run_figure5

# Figure 6: RIS element sweep (Takes ~30 mins)
python -m experiments.run_figure6