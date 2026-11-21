"""One-shot runner for all Figure-3 experiments."""
from __future__ import annotations

import argparse

from experiments.figure3 import main as plot_figure3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",
                   choices=["mahdrl", "mahdrl_no_ris", "maddpg", "cc_mahdrl",
                            "all", "plot"],
                   default="mahdrl")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.algo == "plot":
        plot_figure3()
        return

    if args.algo in ("mahdrl", "all"):
        print("=" * 70)
        print(f"Training MAHDRL ({args.episodes} episodes)")
        print("=" * 70)
        from training.train_mahdrl import train
        train(args.episodes, output_tag="mahdrl", seed=args.seed)
        print()

    if args.algo in ("mahdrl_no_ris", "all"):
        print("=" * 70)
        print(f"Training MAHDRL-no-RIS ({args.episodes} episodes)")
        print("=" * 70)
        from training.train_mahdrl import train
        train(args.episodes, output_tag="mahdrl_no_ris", seed=args.seed,
              disable_ris=True)
        print()

    if args.algo in ("maddpg", "all"):
        print("=" * 70)
        print(f"Training MADDPG ({args.episodes} episodes)")
        print("=" * 70)
        from training.train_maddpg import train_maddpg
        train_maddpg(args.episodes, output_tag="maddpg", seed=args.seed)
        print()

    if args.algo in ("cc_mahdrl", "all"):
        print("=" * 70)
        print(f"Training CC-MAHDRL (M=1 degenerate) ({args.episodes} episodes)")
        print("=" * 70)
        from training.train_cc_mahdrl import train_cc_mahdrl
        train_cc_mahdrl(args.episodes, output_tag="cc_mahdrl", seed=args.seed)
        print()

    plot_figure3()


if __name__ == "__main__":
    main()