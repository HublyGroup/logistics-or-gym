"""Evaluate a trained rl4co (POMO) checkpoint with various inference configs.

Implements Tier 1 from issue #3: free wins on top of an existing checkpoint —
no retraining, just better use of inference-time tricks.

The configs we sweep:
    - greedy:        single start, no augment, greedy decode  (worst case)
    - pomo:          training-default num_starts/num_augment   (the saved val/reward)
    - more-starts-N: more multi-start rollouts                 (~+0.2 to +0.4)
    - augment-64:    8x more symmetric reflections             (~+0.1)
    - sampling-N:    sample N tours per start, take best       (~+0.1 to +0.3)
    - combined:      stack the strongest knobs                 (~+0.4 to +0.6)

Usage:
    uv run --group rl4co python -m training.eval_rl4co \\
        --checkpoint runs/rl4co_mtvrp/final.ckpt \\
        --num-loc 50 --num-instances 1280
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rl4co.envs import MTVRPEnv
from rl4co.models import POMO
from rl4co.utils.ops import unbatchify

# (num_starts, num_augment, decode_type)
# num_starts = None means "auto" (env.get_num_starts(td), which is num_loc for VRP).
# Notes on what's actually tunable at inference time:
#  - `num_starts` is capped by env.get_num_starts(td) (= num_loc for CVRP).
#    Asking for more than that gives duplicates with greedy decode (no gain)
#    but adds genuine diversity with sampling decode.
#  - `num_augment` is fixed by the trained `model.augment` callable
#    (dihedral8 → 8). To change it we'd need to re-init augmentation or
#    retrain. Keeping it at 8 here.
#  - The real free lever is sampling decode: produces more diversity per
#    start, then we take best.
CONFIGS: dict[str, dict] = {
    "greedy": dict(num_starts=1, num_augment=0, decode_type="greedy"),
    "pomo": dict(num_starts=None, num_augment=8, decode_type="greedy"),
    "sampling-50": dict(num_starts=None, num_augment=8, decode_type="sampling"),
    "sampling-200": dict(num_starts=200, num_augment=8, decode_type="sampling"),
    "sampling-500": dict(num_starts=500, num_augment=8, decode_type="sampling"),
}


@torch.no_grad()
def _evaluate_chunk(
    model: POMO,
    env: MTVRPEnv,
    td_chunk,
    num_starts: int | None,
    num_augment: int,
    decode_type: str,
) -> torch.Tensor:
    """Run one inference config on a single chunk; best reward over (aug, start)."""

    td = td_chunk.clone()

    n_start = env.get_num_starts(td) if num_starts is None else num_starts
    n_aug = num_augment if num_augment > 0 else 1

    if num_augment > 1:
        # rl4co's augment is stored as model.augment (StateAugmentation callable)
        td = model.augment(td)

    out = model.policy(
        td, env, phase="test", num_starts=n_start, decode_type=decode_type
    )

    reward = unbatchify(out["reward"], (n_aug, n_start))
    if n_start > 1:
        reward = reward.max(dim=-1).values
    else:
        reward = reward.squeeze(-1)
    if n_aug > 1:
        reward = reward.max(dim=-1).values
    else:
        reward = reward.squeeze(-1)
    return reward.detach().cpu()


@torch.no_grad()
def evaluate_config(
    model: POMO,
    env: MTVRPEnv,
    td_template,
    num_starts: int | None,
    num_augment: int,
    decode_type: str,
    eval_batch_size: int | None = None,
) -> torch.Tensor:
    """Run one inference config over all instances, optionally in chunks.

    High `num_starts` × `num_augment` blows up the effective batch
    (instances × aug × starts). Chunking keeps peak memory bounded so configs
    like sampling-500 stay tractable.
    """
    n_instances = td_template.batch_size[0]
    if eval_batch_size is None or eval_batch_size >= n_instances:
        return _evaluate_chunk(
            model, env, td_template, num_starts, num_augment, decode_type
        )

    rewards = []
    for start in range(0, n_instances, eval_batch_size):
        end = min(start + eval_batch_size, n_instances)
        rewards.append(
            _evaluate_chunk(
                model, env, td_template[start:end],
                num_starts, num_augment, decode_type,
            )
        )
    return torch.cat(rewards, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-loc", type=int, default=50)
    parser.add_argument("--variant", default="cvrp")
    parser.add_argument("--num-instances", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--configs",
        default="greedy,pomo,sampling-50,sampling-200",
        help="Comma-separated list of configs from CONFIGS. "
        "sampling-500 is available but excluded by default (diminishing "
        "returns for ~2.5x the runtime).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Must match the checkpoint's training arch (Tier 2 default 256).",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Process instances in chunks of this size to bound peak memory. "
        "Recommended (e.g. 256) for high num_starts configs like sampling-500.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    env = MTVRPEnv(
        generator_params={"num_loc": args.num_loc, "variant_preset": args.variant}
    )

    print(f"Loading checkpoint from {args.checkpoint} on {device}")
    # Patch out rl4co's broken env-RNG restoration so torch.load doesn't blow up.
    # We load only the state_dict; the env we constructed above is what we use.
    from rl4co.envs.common.base import RL4COEnvBase

    def _setstate_skip_rng(self, state):
        rng = state.pop("rng", None)
        self.__dict__.update(state)
        try:
            if rng is not None:
                self.rng.set_state(rng)
        except Exception:
            pass

    RL4COEnvBase.__setstate__ = _setstate_skip_rng

    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = {
        k: v for k, v in raw["state_dict"].items() if not k.startswith("env.")
    }
    model = POMO(
        env=env,
        batch_size=8,
        train_data_size=8,
        val_data_size=8,
        policy_kwargs={"embed_dim": args.embed_dim, "num_heads": args.num_heads},
    )
    _, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    model.eval().to(device)

    torch.manual_seed(args.seed)
    td_template = env.reset(batch_size=[args.num_instances]).to(device)

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}. Known: {list(CONFIGS)}")

    print(
        f"\nEvaluating on {args.num_instances} instances of "
        f"{args.variant.upper()}-{args.num_loc} (seed={args.seed})"
    )

    results: dict[str, dict] = {}
    for name in requested:
        cfg = CONFIGS[name]
        t0 = time.time()
        rewards = evaluate_config(
            model, env, td_template, eval_batch_size=args.eval_batch_size, **cfg
        )
        elapsed = time.time() - t0
        results[name] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "time": elapsed,
        }
        print(
            f"  {name:<20} mean={results[name]['mean']:.4f}  "
            f"std={results[name]['std']:.3f}  time={elapsed:.1f}s"
        )

    baseline_name = "greedy" if "greedy" in results else requested[0]
    baseline = results[baseline_name]["mean"]

    print()
    print("=" * 78)
    print(
        f"{'config':<22}{'mean':>12}{'std':>10}{'time(s)':>10}"
        f"{'Δ vs ' + baseline_name:>20}"
    )
    print("-" * 78)
    for name, r in results.items():
        delta = (r["mean"] - baseline) / abs(baseline) * 100 if baseline else 0
        print(
            f"{name:<22}{r['mean']:>12.4f}{r['std']:>10.3f}"
            f"{r['time']:>10.1f}{delta:>+18.2f}%"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
