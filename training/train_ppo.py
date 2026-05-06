"""Train MaskablePPO on HeterogeneousCVRP.

This is a *baseline*. It's not expected to beat purpose-built routing solvers.
The goal is to validate the env end-to-end and produce a comparison floor for
rl4co-based agents (issue #2).

Usage:
    uv run python -m training.train_ppo --total-timesteps 200000
    uv run python -m training.train_ppo --total-timesteps 200000 --n-envs 8 --tensorboard
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from logistics_or_gym.envs.HeterogeneousCVRP import HeterogeneousCVRP
from training.wrappers import FlatMaskableWrapper


def make_env(
    n_nodes: int,
    n_vehicles: int,
    objective: str,
    demand_dist: str,
    capacities: list[float] | None = None,
    speeds: list[float] | None = None,
):
    def _make():
        env = HeterogeneousCVRP(
            n_nodes=n_nodes,
            n_vehicles=n_vehicles,
            objective=objective,
            demand_dist=demand_dist,
            capacities=capacities,
            speeds=speeds,
        )
        return Monitor(FlatMaskableWrapper(env))

    return _make


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--n-vehicles", type=int, default=3)
    parser.add_argument(
        "--objective", choices=["min_max", "min_sum"], default="min_sum"
    )
    parser.add_argument(
        "--demand-dist", choices=["uniform", "discrete"], default="uniform"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/ppo_hcvrp")
    )
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help='"auto" (GPU if available), "cuda", "cuda:0", "cpu"',
    )
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)

    factory = make_env(
        n_nodes=args.n_nodes,
        n_vehicles=args.n_vehicles,
        objective=args.objective,
        demand_dist=args.demand_dist,
    )
    train_env = DummyVecEnv([factory for _ in range(args.n_envs)])
    eval_env = DummyVecEnv([factory])

    tb_log = str(args.run_dir / "tb") if args.tensorboard else None

    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        seed=args.seed,
        tensorboard_log=tb_log,
        device=args.device,
        verbose=1,
    )
    print(f"Training on device: {model.device}")

    callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(args.run_dir / "best"),
        log_path=str(args.run_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(args.run_dir / "final")

    final_mean, final_std = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    print(
        f"\nFinal eval over {args.n_eval_episodes} episodes: "
        f"{final_mean:.3f} ± {final_std:.3f}"
    )


if __name__ == "__main__":
    main()
