"""Train POMO / Attention Model on rl4co's MTVRPEnv.

MTVRPEnv covers paper 1's five variants (Capacity, Open Route, Backhaul,
Duration Limit, Time Window) and combinations of them. POMO is the dominant
RL approach for routing problems and routinely closes the gap to ~3% of optimal.

Tier 2 features (issue #4):
- LR scheduler: MultiStepLR by default with the POMO-paper milestones.
- Configurable embedding_dim / num_heads (policy capacity).
- Resume from a previous checkpoint.
- Periodic + best-val-reward checkpointing.
- LearningRateMonitor logged to TensorBoard.

Usage:
    # Default Tier 2 recipe: 200 epochs, embed_dim=256, LR 1e-4 → 1e-5 → 1e-6
    uv run --group rl4co python -m training.train_rl4co \\
        --variant cvrp --num-loc 50 --epochs 200 --embed-dim 256 \\
        --lr-scheduler multistep --lr-milestones 100,175 --lr-gamma 0.1

    # Resume training from a checkpoint (must match arch — same embed_dim etc)
    uv run --group rl4co python -m training.train_rl4co --resume-from runs/.../last.ckpt

    # Multi-task across all paper-1 variants
    uv run --group rl4co python -m training.train_rl4co --variant all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# --- Compatibility patches needed when resuming from checkpoints ---
# rl4co serializes its env into Lightning's hyperparameters. Two PyTorch 2.6+ /
# rl4co interactions break the resume path:
#   1. torch.load defaults to weights_only=True, blocking unpickling of MTVRPEnv
#   2. rl4co's RL4COEnvBase.__setstate__ has a buggy RNG restore that raises
#      `TypeError: RNG state must be a torch.ByteTensor` even with weights_only=False
# We patch both before importing anything that might trigger torch.load.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # Force-override; Lightning explicitly passes weights_only=True
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from rl4co.envs import MTVRPEnv  # noqa: E402
from rl4co.envs.common.base import RL4COEnvBase  # noqa: E402
from rl4co.models import POMO, AttentionModel  # noqa: E402
from rl4co.utils.trainer import RL4COTrainer  # noqa: E402


def _setstate_skip_rng(self, state):
    rng = state.pop("rng", None)
    self.__dict__.update(state)
    try:
        if rng is not None:
            self.rng.set_state(rng)
    except Exception:
        pass


RL4COEnvBase.__setstate__ = _setstate_skip_rng

VARIANT_PRESETS = [
    "cvrp",
    "ovrp",
    "vrpb",
    "vrpl",
    "vrptw",
    "ovrptw",
    "ovrpb",
    "ovrpl",
    "vrpbl",
    "vrpbtw",
    "vrpltw",
    "ovrpbl",
    "ovrpbtw",
    "ovrpltw",
    "vrpbltw",
    "ovrpbltw",
    "all",
]


def parse_milestones(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    # Problem / variant
    parser.add_argument(
        "--variant",
        choices=VARIANT_PRESETS,
        default="cvrp",
        help='Variant preset. "all" trains on a random mix every batch.',
    )
    parser.add_argument("--num-loc", type=int, default=50)

    # Model
    parser.add_argument(
        "--model",
        choices=["pomo", "am"],
        default="pomo",
        help="POMO (multi-start, dominant for routing) or vanilla Attention Model.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Policy embedding dim. Tier 2 recipe uses 256 (~5M params).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads in encoder/decoder.",
    )

    # Training scale
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Per-step batch (instances)."
    )
    parser.add_argument(
        "--train-data-size",
        type=int,
        default=100_000,
        help="Instances per epoch (auto-generated).",
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=10_000,
        help="Instances for validation per epoch.",
    )

    # Optimizer + LR schedule
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "multistep", "cosine"],
        default="multistep",
    )
    parser.add_argument(
        "--lr-milestones",
        type=str,
        default="100,175",
        help='Epoch milestones for MultiStepLR (comma-separated).',
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="Decay factor at each milestone (or final factor for cosine).",
    )

    # Runtime
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 is fine for on-the-fly generation.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs. Set to 0 for CPU.",
    )

    # Checkpointing / I/O
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rl4co_mtvrp"))
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint to resume training from (architecture must match).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save a periodic checkpoint every N epochs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Env
    generator_params = {"num_loc": args.num_loc}
    if args.variant != "all":
        generator_params["variant_preset"] = args.variant
    env = MTVRPEnv(generator_params=generator_params)

    # LR scheduler config
    lr_scheduler = None
    lr_scheduler_kwargs: dict = {}
    if args.lr_scheduler == "multistep":
        lr_scheduler = "MultiStepLR"
        lr_scheduler_kwargs = {
            "milestones": parse_milestones(args.lr_milestones),
            "gamma": args.lr_gamma,
        }
    elif args.lr_scheduler == "cosine":
        lr_scheduler = "CosineAnnealingLR"
        lr_scheduler_kwargs = {
            "T_max": args.epochs,
            "eta_min": args.learning_rate * args.lr_gamma,
        }

    # Model
    common_kwargs = dict(
        env=env,
        batch_size=args.batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        optimizer_kwargs={"lr": args.learning_rate},
        policy_kwargs={
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
        },
        dataloader_num_workers=args.num_workers,
    )
    if lr_scheduler is not None:
        common_kwargs["lr_scheduler"] = lr_scheduler
        common_kwargs["lr_scheduler_kwargs"] = lr_scheduler_kwargs

    if args.model == "pomo":
        model = POMO(**common_kwargs)
    else:
        model = AttentionModel(**common_kwargs)

    # Callbacks
    ckpt_dir = args.run_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}-val_reward={val/reward:.4f}",
            monitor="val/reward",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="periodic-{epoch:03d}",
            every_n_epochs=args.checkpoint_every,
            save_top_k=-1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    accelerator = "gpu" if torch.cuda.is_available() and args.devices > 0 else "cpu"
    devices = args.devices if accelerator == "gpu" else 1

    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(args.run_dir),
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )

    # Param count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Training {args.model.upper()} on MTVRP({args.variant}, num_loc={args.num_loc}) "
        f"on {accelerator}\n"
        f"  embed_dim={args.embed_dim}, num_heads={args.num_heads}, params={n_params/1e6:.2f}M\n"
        f"  lr={args.learning_rate}, scheduler={args.lr_scheduler}"
        + (
            f" milestones={lr_scheduler_kwargs.get('milestones')}"
            f" gamma={lr_scheduler_kwargs.get('gamma')}"
            if args.lr_scheduler == "multistep"
            else ""
        )
        + f"\n  batch_size={args.batch_size}, epochs={args.epochs}, "
        f"train_data={args.train_data_size}/epoch"
    )

    trainer.fit(
        model,
        ckpt_path=str(args.resume_from) if args.resume_from else None,
    )
    trainer.save_checkpoint(args.run_dir / "final.ckpt")
    print(f"\nFinal checkpoint saved to {args.run_dir / 'final.ckpt'}")
    print(f"Best/periodic checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
