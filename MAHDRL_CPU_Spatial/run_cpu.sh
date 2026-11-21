#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v python3.11 >/dev/null || ! rpm -q python3.11-pip >/dev/null; then
  sudo dnf install -y python3.11 python3.11-pip
fi
python3.11 -m venv .venv-cpu
py=.venv-cpu/bin/python
"$py" -m pip install --upgrade pip
"$py" -m pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu
"$py" -m pip install -r requirements.txt
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
case "${1:-full}" in
  full) "$py" -u run_fig3_experiment.py --algo all --episodes 400 --seed 7 ;;
  sweeps) for n in 4 5 6; do "$py" -u -m experiments.run_figure"$n"; done ;;
  *) echo 'Use full or sweeps'; exit 2 ;;
esac
