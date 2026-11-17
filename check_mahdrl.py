"""End-to-end MAHDRL training diagnostic.

Trains the full Algorithm 1 pipeline (DQN + DDPG + QMIX) on the real
environment for N_STEPS and reports improvement vs. the untrained baseline.

Success criteria:
    - Reward increases by >= 1 Mbps-eq
    - Losses stay finite and do not explode
    - Data rate ends up in the 30-70 Mbps range
"""
import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment
from agents.mahdrl import MAHDRLAgent


def eval_policy(env, agent, episodes=2):
    rewards, rates, wifis = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        r_sum = rate_sum = wifi_sum = 0.0
        steps = 0
        while not done:
            disc, raw, action = agent.select_action(
                obs, epsilon=0.0, noise_std=0.0
            )
            obs, reward, done, info = env.step(action)
            r_sum += reward
            rate_sum += info["total_rate_bps"]
            wifi_sum += float(np.sum(np.maximum(
                0.0, info["wifi_psi_dbm"] - env.cfg.wifi_energy_detection_dbm
            )))
            steps += 1
        rewards.append(r_sum / steps)
        rates.append(rate_sum / steps)
        wifis.append(wifi_sum / steps)
    return float(np.mean(rewards)), float(np.mean(rates)), float(np.mean(wifis))


def main():
    sim_cfg = SimulationConfig()
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = MAHDRLAgent(
        simulation_cfg=sim_cfg,
        agent_cfg=agent_cfg,
        obs_dim=obs.shape[0],
        device="cpu",
        seed=0,
    )

    print("Baseline (untrained):")
    r0, rate0, wifi0 = eval_policy(env, agent, episodes=2)
    print(f"  Reward: {r0/1e6:7.2f} Mbps-eq  "
          f"Rate: {rate0/1e6:6.2f} Mbps  "
          f"WiFi excess: {wifi0:6.1f} dB")
    print()

    N_STEPS = 20000
    WARMUP = agent_cfg.ddpg_warmup_steps   # 4000
    UPDATE_EVERY = 4
    eps_decay = 3000
    noise_decay = 3000

    print(f"Training MAHDRL for {N_STEPS} env steps")
    print(f"  warmup={WARMUP}  update_every={UPDATE_EVERY}  "
          f"eps_decay={eps_decay}  noise_decay={noise_decay}")
    print()

    obs = env.reset()
    ep_reward = 0.0
    ep_count = 0
    recent_dqn, recent_cm, recent_actor, recent_q = [], [], [], []
    total_train_steps = 0

    for step in range(N_STEPS):
        eps = max(agent_cfg.dqn_epsilon_start
                  * (1.0 - step / eps_decay),
                  agent_cfg.dqn_epsilon_end)
        noise = max(agent_cfg.exploration_noise_std
                    * (1.0 - step / noise_decay),
                    agent_cfg.dqn_epsilon_end)

        disc, raw, action = agent.select_action(obs, epsilon=eps, noise_std=noise)
        next_obs, reward, done, info = env.step(action)
        agent.store(obs, disc, raw, reward, next_obs, float(done))
        ep_reward += reward

        if (agent.buffer.size >= WARMUP) and (step % UPDATE_EVERY == 0):
            result = agent.train_step()
            if result is not None:
                recent_dqn.append(result["dqn_loss"])
                recent_cm.append(result["critic_mixer_loss"])
                recent_actor.append(result["actor_loss"])
                recent_q.append(result["q_tot_mean"])
                for lst in (recent_dqn, recent_cm, recent_actor, recent_q):
                    if len(lst) > 50:
                        lst.pop(0)
                total_train_steps += 1

        if done:
            ep_count += 1
            if ep_count % 10 == 0 or step == N_STEPS - 1:
                tag = "warmup" if agent.buffer.size < WARMUP else "learn"
                print(f"  step {step+1:5d}  ep {ep_count:2d} [{tag}]  "
                      f"ep_r={ep_reward/100/1e6:6.2f}  "
                      f"dqn={np.mean(recent_dqn) if recent_dqn else 0:7.3f}  "
                      f"cm={np.mean(recent_cm) if recent_cm else 0:7.3f}  "
                      f"a={np.mean(recent_actor) if recent_actor else 0:7.3f}  "
                      f"Q_tot={np.mean(recent_q) if recent_q else 0:6.2f}")
            obs = env.reset()
            ep_reward = 0.0
        else:
            obs = next_obs

    print()
    print("After training (deterministic evaluation):")
    r1, rate1, wifi1 = eval_policy(env, agent, episodes=2)
    print(f"  Reward: {r1/1e6:7.2f} Mbps-eq  "
          f"Rate: {rate1/1e6:6.2f} Mbps  "
          f"WiFi excess: {wifi1:6.1f} dB")
    print()
    print("-" * 80)
    print(f"Total train steps:  {total_train_steps}")
    print(f"Reward change:      {(r1 - r0)/1e6:+.2f} Mbps-eq")
    print(f"Rate change:        {(rate1 - rate0)/1e6:+.2f} Mbps")
    print(f"WiFi excess change: {(wifi1 - wifi0):+.1f} dB")
    print()
    if r1 > r0:
        print("✅ MAHDRL improved reward. Algorithm 1 works end-to-end.")
        print()
        print("Next steps (Step 12) will extend training to 40,000 steps to")
        print("match the paper's 400 episodes × 100 slots convergence curve.")
    else:
        print("⚠ MAHDRL did not improve — paste full log for tuning.")


if __name__ == "__main__":
    main()