"""Diagnostic: verify QMIX monotonicity (paper Eq. 35).

Prints a table showing Q_tot as Q_1 increases from 0.0 to 1.0 by 0.1, for
M = 1, 2, 3, 4 agents. Q_tot must increase monotonically in every case.
"""
import numpy as np

from agents.mixing_network import QMixer


def main():
    state_dim = 80
    state = np.zeros(state_dim, dtype=np.float32)   # fixed reference state

    print("=" * 80)
    print("QMIX monotonicity check (paper Eq. 35): ∂Q_tot/∂Q_m ≥ 0 for all m")
    print("=" * 80)
    print()

    for M in (1, 2, 3, 4):
        mixer = QMixer(num_agents=M, state_dim=state_dim, device="cpu")
        print(f"M = {M} agent(s):")
        print(f"  {'Q_1':>5}  {'Q_tot':>12}  {'ΔQ_tot':>12}")
        prev = None
        all_monotone = True
        for q1_val in np.linspace(0.0, 1.0, 11):
            qs = np.full(M, 0.5, dtype=np.float32)
            qs[0] = q1_val
            q_tot = float(mixer.compute_q_tot(qs, state))
            delta = "—" if prev is None else f"{q_tot - prev:+.6f}"
            marker = ""
            if prev is not None and q_tot < prev - 1e-6:
                marker = "  ❌ NON-MONOTONE"
                all_monotone = False
            print(f"  {q1_val:5.1f}  {q_tot:12.6f}  {delta:>12}{marker}")
            prev = q_tot
        print(f"  → {'✅ monotone' if all_monotone else '❌ FAILED'}")
        print()

    print("=" * 80)
    print("Argument (paper Eq. 34):")
    print("  If Q_tot is monotone increasing in each Q_m, then")
    print("      argmax Q_tot  ==  {argmax Q_1, argmax Q_2, ...}")
    print("  In other words, greedy per-agent policies combine into")
    print("  the globally optimal joint policy.")
    print("=" * 80)


if __name__ == "__main__":
    main()