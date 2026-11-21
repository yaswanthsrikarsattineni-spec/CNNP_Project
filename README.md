# MAHDRL for RIS-assisted wireless networks

This project uses reinforcement learning to schedule wireless users, steer transmit beams, and control a reconfigurable intelligent surface (RIS). The experiment asks whether the controller can improve throughput while keeping user delay and interference to nearby Wi-Fi devices low.

The CPU implementation is in [MAHDRL_CPU_Spatial](MAHDRL_CPU_Spatial/). It adapts the multi-agent hierarchical deep reinforcement learning (MAHDRL) method from Chen et al. to a **single simulated base station**. The goal is to compare trends with the paper. This implementation cannot establish how well multiple base stations coordinate.

## Run Figure 3 on Red Hat Enterprise Linux 9

You need Python 3.11, internet access for the first dependency installation, and a CPU. The launcher installs the CPU build of PyTorch. From a new terminal, run:

```bash
# For a fresh checkout
cd ~/Downloads
git clone https://github.com/yaswanthsrikarsattineni-spec/CNNP_Project.git
```

If the repository is already in Downloads, start here:

```bash
cd ~/Downloads/CNNP_Project/MAHDRL_CPU_Spatial
sudo dnf install -y python3.11 python3.11-pip
set -o pipefail
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV \
  PATH=/usr/bin:/bin bash run_cpu.sh full 2>&1 | tee training.log
```

The clean environment lets the launcher use Red Hat's Python even when a CAD tool has configured its own Python runtime. The launcher creates `.venv-cpu`, installs dependencies, and runs **four methods for 400 episodes each**, requesting seed 7. Each episode contains 100 simulation slots. A repeat run writes to the same log, graph, and checkpoint paths, so copy any results you want to retain first.

The run finishes when the terminal prints `[figure3] Saved: results/graphs/figure3.png`. Open the new graph from the model directory:

```bash
xdg-open results/graphs/figure3.png
```

### Recover an environment created with OSS CAD Suite Python

If the error mentions `tabbypy3`, `No module named encodings`, or a missing `pip` inside an existing environment, run this from the model directory after installing the RHEL packages above:

```bash
(
  set -e
  unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH VIRTUAL_ENV
  export PATH=/usr/bin:/bin
  if [ -d .venv-cpu ]; then
    mv .venv-cpu ".venv-cpu.backup.$(date +%s)"
  fi
  /usr/bin/python3.11 -m venv --without-pip .venv-cpu
  /usr/bin/python3.11 -m pip --python .venv-cpu/bin/python install pip
  bash run_cpu.sh full
)
```

## How one simulation slot works

1. **Observe.** The controller receives local information about the wireless channels, queues, recent rates, and Wi-Fi interference.
2. **Choose users.** The upper-level Deep Q-Network (DQN) decides which users to serve. Selection is a discrete decision.
3. **Allocate resources.** The lower-level actor uses Deep Deterministic Policy Gradient (DDPG) to choose continuous controls for transmission and RIS phases. Changing a phase changes how reflected signals combine with the direct signal.
4. **Calculate rates.** For each scheduled user, the simulator calculates the signal-to-interference-plus-noise ratio (SINR). Each simultaneous spatial stream uses the full channel bandwidth. Its rate in bits per second is `bandwidth_hz * log2(1 + SINR)`. Summing the user rates gives the base-station rate.
5. **Update and learn.** The environment serves queued traffic, adds arrivals, and evaluates reward using rate, delay, interference, and backlog. The trainer stores the transition in replay memory and samples earlier transitions to update the networks.

| Code inside the model directory | What to explain in a walkthrough |
|---|---|
| `config.py` | Physical settings, queue limits, and learning parameters |
| `environment/simulation.py` | How a scheduling decision affects users and Wi-Fi devices during one slot |
| `environment/data_rate.py` | How SINR becomes throughput |
| `agents/mahdrl.py` | How the discrete and continuous decisions interact |
| `training/train_mahdrl.py` | How the simulator, replay buffer, and network updates form a training loop |
| `run_fig3_experiment.py` | How the four methods are launched and compared |

## Methods and default settings

| Method | Purpose |
|---|---|
| MAHDRL | Hierarchical user selection and continuous radio control with RIS |
| MAHDRL without RIS | Measures the effect of removing the reflected signal paths |
| MADDPG | Baseline with continuous user-selection decisions |
| CC-MAHDRL | Centralized-critic comparison. With one base station, the comparison cannot isolate the benefit of multi-agent mixing. |

The default deployment uses a 300 m × 300 m area, a 5 GHz carrier, 40 MHz bandwidth, 10 base-station antennas, 12 users, and one 20-element RIS. Four Wi-Fi access points each serve eight devices. The configured minimum user rate is 3 Mbps, the delay limit is 30 ms, and the Wi-Fi detection threshold is −62 dBm.

## Read the outputs

Paths below are relative to `MAHDRL_CPU_Spatial/`.

| Output | Interpretation |
|---|---|
| `results/logs/*_training.csv` | One row per episode for each method. Divide `episode_rate_bps` by one million for Mbps. |
| `results/graphs/figure3.png` | Reward and throughput during training |
| `results/checkpoints/` | Saved model weights |
| `results/logs/summary.csv` | Separately collected first-50 and last-50 episode means for seeds 7, 17, and 27. The single-seed launcher produces the raw training CSVs. |

**Reading the three-seed summary.** Compare rows with the same seed and window. For MAHDRL, the average base-station rate increased from the first 50 to the last 50 episodes for each seed:

| Seed | First 50 episodes (Mbps) | Last 50 episodes (Mbps) |
|---|---:|---:|
| 7 | 182.98 | 191.12 |
| 17 | 186.47 | 192.76 |
| 27 | 180.69 | 187.65 |

The throughput increase came with higher mean latency for MAHDRL in all three runs. The late throughput ordering against MADDPG varied by seed. MAHDRL exceeded MADDPG for seed 7, while MADDPG exceeded MAHDRL for seeds 17 and 27. These training averages support a modest upward throughput trend. Generalization requires evaluation on fresh scenarios.

The Wi-Fi excess column sums positive deviations above the detection threshold across devices and averages that sum over slots. Lower values indicate less aggregate excess. Its unit is dB and it cannot be read as received power in dBm. The training log's reliability field records the fraction of users below the queue limit at the final slot of the episode.

## Run the parameter sweeps

From the model directory, use the same CPU launcher:

```bash
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV \
  PATH=/usr/bin:/bin bash run_cpu.sh sweeps
```

Figure 4 varies user count to study per-user rate, latency, and Wi-Fi interference. Figure 5 varies Wi-Fi access-point count to study interference and queue reliability. Figure 6 varies RIS element count to study base-station throughput and Wi-Fi interference. The scripts write sweep CSVs into `results/logs/` and plots into `results/graphs/`. Use outputs from the same run when comparing a graph with its underlying data.

## Reference

Qimei Chen, Ruixue Li, Xiaoxia Xu, Jing Wu, Hao Jiang, and Meikang Qiu. “Human-Aware Dynamic Hierarchical Network Control for Distributed Metaverse Services.” *IEEE Journal on Selected Areas in Communications*, vol. 42, no. 3, pp. 629–642, 2024. [DOI 10.1109/JSAC.2023.3345399](https://doi.org/10.1109/JSAC.2023.3345399)
