"""End-to-end DDPG training with warmup + reduced update frequency.

Training recipe (all standard DDPG/TD3 practice):
    - Warmup 3000 env steps before any gradient update.
    - Update critic every 4 env steps.
    - Update actor every 2 critic updates.
    - Clip target Q to ±30.

Success criterion: reward after training > baseline (untrained).
"""
import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment
from agents.ddpg_actor import DDPGActor
from agents.ddpg_critic import DDPGCritic
from agents.replay_buffer import ReplayBuffer


def make_discrete_action(K, N_ant):
    disc = np.zeros(K, dtype=np.float32)
    disc[: min(N_ant, K)] = 1.0
    return disc


def eval_policy(env, actor, disc, episodes=2):
    rewards, rates, wifis = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        r_sum = rate_sum = wifi_sum = 0.0
        steps = 0
        while not done:
            action = actor.select_action(obs, disc, noise_std=0.0)
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
    cfg = SimulationConfig()
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(cfg)
    obs = env.reset()

    p_tx_w = 10.0 ** ((cfg.max_bs_power_dbm - 30.0) / 10.0)
    actor = DDPGActor(
        obs_dim=obs.shape[0], num_users=cfg.users_per_bs,
        num_ant=cfg.antennas_per_bs, num_ris=cfg.ris_per_bs,
        ris_elements=cfg.ris_elements, max_power_w=p_tx_w,
        agent_cfg=agent_cfg, device="cpu", seed=0,
    )
    critic = DDPGCritic(
        obs_dim=obs.shape[0], num_users=cfg.users_per_bs,
        continuous_dim=actor.output_dim,
        agent_cfg=agent_cfg, device="cpu",
    )
    buffer = ReplayBuffer(
        capacity=agent_cfg.replay_capacity,
        obs_dim=obs.shape[0],
        action_shape=(cfg.users_per_bs + actor.output_dim,),
        seed=0,
    )
    disc = make_discrete_action(cfg.users_per_bs, cfg.antennas_per_bs)

    print("Baseline (untrained):")
    r0, rate0, wifi0 = eval_policy(env, actor, disc, episodes=2)
    print(f"  Reward: {r0/1e6:7.2f} Mbps-eq  "
          f"Rate: {rate0/1e6:6.2f} Mbps  "
          f"WiFi excess: {wifi0:6.1f} dB")
    print()

    N_TRAIN = 15000
    ACTOR_EVERY_CRITIC = 2
    noise_decay = 5000
    print(f"Training DDPG for {N_TRAIN} env steps")
    print(f"  warmup={agent_cfg.ddpg_warmup_steps}  "
          f"update_every={agent_cfg.ddpg_update_every}  "
          f"actor_lr={agent_cfg.actor_learning_rate}  "
          f"target_clip=±{agent_cfg.ddpg_target_clip}")
    print()

    obs = env.reset()
    ep_reward = 0.0
    ep_count = 0
    recent_c, recent_a, recent_q = [], [], []
    total_critic_updates = 0
    total_actor_updates = 0

    for step in range(N_TRAIN):
        eps = max(agent_cfg.exploration_noise_std * (1.0 - step / noise_decay),
                  agent_cfg.dqn_epsilon_end)
        raw, action = actor.select_action_and_raw(obs, disc, noise_std=eps)
        next_obs, reward, done, info = env.step(action)

        buffer.add(obs, np.concatenate([disc, raw]), reward, next_obs, float(done))
        ep_reward += reward

        # Warmup + throttled updates.
        can_learn = (buffer.size >= agent_cfg.ddpg_warmup_steps)
        do_update = can_learn and (step % agent_cfg.ddpg_update_every == 0)

        if do_update:
            batch = buffer.sample(agent_cfg.batch_size)
            c_info = critic.update(actor, batch)
            recent_c.append(c_info["critic_loss"])
            recent_q.append(c_info["q_abs_mean"])
            if len(recent_c) > 50:
                recent_c.pop(0)
                recent_q.pop(0)
            total_critic_updates += 1

            if total_critic_updates % ACTOR_EVERY_CRITIC == 0:
                a_info = actor.update(critic, batch)
                recent_a.append(a_info["actor_loss"])
                if len(recent_a) > 50:
                    recent_a.pop(0)
                total_actor_updates += 1

        if done:
            ep_count += 1
            if ep_count % 10 == 0 or step == N_TRAIN - 1:
                tag = "WARMUP" if not can_learn else "learn"
                print(f"  step {step+1:5d}  ep {ep_count:2d} [{tag}]  "
                      f"ep_r={ep_reward/100/1e6:6.2f}  "
                      f"c_loss={np.mean(recent_c) if recent_c else 0:7.3f}  "
                      f"a_loss={np.mean(recent_a) if recent_a else 0:7.3f}  "
                      f"|Q|={np.mean(recent_q) if recent_q else 0:5.1f}")
            obs = env.reset()
            ep_reward = 0.0
        else:
            obs = next_obs

    print()
    print("After training (deterministic evaluation):")
    r1, rate1, wifi1 = eval_policy(env, actor, disc, episodes=2)
    print(f"  Reward: {r1/1e6:7.2f} Mbps-eq  "
          f"Rate: {rate1/1e6:6.2f} Mbps  "
          f"WiFi excess: {wifi1:6.1f} dB")
    print()
    print("-" * 80)
    print(f"Critic updates:        {total_critic_updates}")
    print(f"Actor updates:         {total_actor_updates}")
    print(f"Reward change:         {(r1 - r0)/1e6:+.2f} Mbps-eq")
    print(f"Rate change:           {(rate1 - rate0)/1e6:+.2f} Mbps")
    print(f"WiFi excess change:    {(wifi1 - wifi0):+.1f} dB")
    print()
    if r1 > r0:
        print("✅ DDPG improved reward after training.")
    else:
        print("⚠ DDPG still not improving. Diagnostics to paste in the reply:")
        print("   - Full training log above")
        print("   - c_loss trend (should DECREASE, not oscillate)")
        print("   - |Q| magnitude (should stay < target_clip, not grow)")
    print()
    print("Note: fixed user selection caps max reward. Full MAHDRL (Step 11)")
    print("      learns user selection too and will go much higher.")


if __name__ == "__main__":
    main()