"""Random-policy baseline. Picks a uniformly-random valid action each step.

This is the floor any learned agent must beat. Run before training to sanity-
check the reward distribution and after training to compute the gap.

Usage:
    uv run python -m training.random_baseline --n-episodes 100 --objective min_sum
"""

from __future__ import annotations

import argparse

import numpy as np

from logistics_or_gym.envs.HeterogeneousCVRP import HeterogeneousCVRP


def run(
    n_episodes: int,
    n_nodes: int,
    n_vehicles: int,
    objective: str,
    demand_dist: str,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    env = HeterogeneousCVRP(
        n_nodes=n_nodes,
        n_vehicles=n_vehicles,
        objective=objective,
        demand_dist=demand_dist,
    )

    returns = []
    episode_lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_return = 0.0
        steps = 0
        while True:
            mask = obs["action_mask"]
            valid = np.argwhere(mask == 1)
            if len(valid) == 0:
                break
            choice = valid[rng.integers(0, len(valid))]
            obs, reward, terminated, truncated, _ = env.step(choice)
            ep_return += float(reward)
            steps += 1
            if terminated or truncated:
                break
        returns.append(ep_return)
        episode_lengths.append(steps)

    returns = np.array(returns)
    episode_lengths = np.array(episode_lengths)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_length": float(episode_lengths.mean()),
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--n-vehicles", type=int, default=3)
    parser.add_argument(
        "--objective", choices=["min_max", "min_sum"], default="min_sum"
    )
    parser.add_argument(
        "--demand-dist", choices=["uniform", "discrete"], default="uniform"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats = run(
        n_episodes=args.n_episodes,
        n_nodes=args.n_nodes,
        n_vehicles=args.n_vehicles,
        objective=args.objective,
        demand_dist=args.demand_dist,
        seed=args.seed,
    )

    print(f"Random baseline | {args.n_episodes} eps | objective={args.objective}")
    print(f"  mean return:   {stats['mean_return']:.3f} ± {stats['std_return']:.3f}")
    print(
        f"  min / max:     {stats['min_return']:.3f} / {stats['max_return']:.3f}"
    )
    print(f"  mean length:   {stats['mean_length']:.1f} steps")


if __name__ == "__main__":
    main()
