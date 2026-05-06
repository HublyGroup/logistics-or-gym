"""Live viewer that plots the latest policy's best tour on a fixed instance.

Watches a checkpoint directory; when `last.ckpt` updates (every val epoch),
reloads the policy, runs inference on a fixed test instance, and re-renders
the tour. Use during training to watch the policy improve in real time.

Two modes:
- `--display` (default if a display is available): live matplotlib window.
- `--save-frames DIR`: writes one PNG per detected checkpoint update.
   Use both together for live-watching with a recording.

Usage:
    # Live during the running training in tmux session rl4co-t2
    uv run --group rl4co python -m training.live_viewer \\
        --checkpoint-dir runs/rl4co_mtvrp_t2/checkpoints \\
        --num-loc 50 --embed-dim 256 --refresh-interval 30

    # Headless: just dump frames as the policy improves
    uv run --group rl4co python -m training.live_viewer \\
        --checkpoint-dir runs/rl4co_mtvrp_t2/checkpoints \\
        --num-loc 50 --embed-dim 256 \\
        --save-frames runs/rl4co_mtvrp_t2/frames --no-display
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

import numpy as np
import torch
from rl4co.envs import MTVRPEnv
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models import POMO


def patch_env_setstate():
    """Skip rl4co's broken RNG-state restore on env unpickle."""

    def _setstate_skip_rng(self, state):
        rng = state.pop("rng", None)
        self.__dict__.update(state)
        try:
            if rng is not None:
                self.rng.set_state(rng)
        except Exception:
            pass

    RL4COEnvBase.__setstate__ = _setstate_skip_rng


def load_policy(ckpt_path, env, device, embed_dim, num_heads):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {
        k: v for k, v in raw["state_dict"].items() if not k.startswith("env.")
    }
    epoch = raw.get("epoch", None)

    model = POMO(
        env=env,
        batch_size=8,
        train_data_size=8,
        val_data_size=8,
        policy_kwargs={"embed_dim": embed_dim, "num_heads": num_heads},
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, epoch


@torch.no_grad()
def get_best_tour(model, env, td):
    n_start = env.get_num_starts(td)
    out = model.policy(
        td, env, phase="test", num_starts=n_start, decode_type="greedy"
    )
    rewards = out["reward"]
    actions = out["actions"]
    best_idx = int(rewards.argmax())
    return actions[best_idx].cpu().numpy(), float(rewards[best_idx])


def decompose_trips(actions: np.ndarray) -> list[list[int]]:
    trips: list[list[int]] = []
    current = [0]
    for a in actions:
        a = int(a)
        current.append(a)
        if a == 0 and len(current) > 2:
            trips.append(current)
            current = [0]
    if len(current) > 1:
        if current[-1] != 0:
            current.append(0)
        trips.append(current)
    return trips


def compute_trip_stats(trips, locs, demand, capacity):
    stats = []
    for trip in trips:
        customers = [n for n in trip[1:-1] if n != 0]
        demand_sum = float(sum(float(demand[c]) for c in customers))
        dist = 0.0
        for j in range(len(trip) - 1):
            a, b = trip[j], trip[j + 1]
            dist += float(np.linalg.norm(locs[a] - locs[b]))
        stats.append(
            {
                "n_customers": len(customers),
                "demand": demand_sum,
                "distance": dist,
                "load_pct": (100 * demand_sum / capacity) if capacity > 0 else 0,
            }
        )
    return stats


def render_tour(fig, ax, td, locs, demand, actions, reward, epoch=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ax.clear()

    capacity = float(td["vehicle_capacity"][0].item())
    speed = float(td["speed"][0].item())
    open_route = bool(td["open_route"][0].item())
    distance_limit = float(td["distance_limit"][0].item())

    depot = locs[0]
    ax.scatter(
        depot[0], depot[1],
        marker="s", s=300, c="red", edgecolors="black",
        linewidths=1.5, zorder=5,
    )

    customer_locs = locs[1:]
    customer_demand = demand[1:]
    max_d = max(float(customer_demand.max()), 1e-9)
    sizes = 30 + 100 * (customer_demand / max_d)
    ax.scatter(
        customer_locs[:, 0], customer_locs[:, 1],
        s=sizes, c="lightgray", edgecolors="black", zorder=3,
    )

    trips = decompose_trips(actions)
    stats = compute_trip_stats(trips, locs, demand, capacity)

    cmap = plt.get_cmap("tab10")
    for i, trip in enumerate(trips):
        color = cmap(i % 10)
        for j in range(len(trip) - 1):
            a, b = trip[j], trip[j + 1]
            xa, ya = locs[a]
            xb, yb = locs[b]
            ax.annotate(
                "",
                xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=1.6,
                    alpha=0.75, shrinkA=4, shrinkB=4,
                ),
                zorder=4,
            )

    n_customers = len(locs) - 1
    total_demand = float(demand[1:].sum())

    spec_lines = [
        f"capacity:       {capacity:.2f}",
        f"speed:          {speed:.2f}",
        f"customers:      {n_customers}",
        f"total demand:   {total_demand:.2f}",
    ]
    if open_route:
        spec_lines.append("open route:     yes")
    if distance_limit > 0 and distance_limit < 1e6:
        spec_lines.append(f"distance limit: {distance_limit:.2f}")

    handles = [
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="red",
            markeredgecolor="black", markersize=10, label="depot",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="lightgray",
            markeredgecolor="black", markersize=8, label=f"customer (sized by demand)",
        ),
        Patch(facecolor="none", edgecolor="none", label=""),
        Patch(facecolor="none", edgecolor="none", label="── instance ──"),
    ]
    for line in spec_lines:
        handles.append(Patch(facecolor="none", edgecolor="none", label=line))
    handles.append(Patch(facecolor="none", edgecolor="none", label=""))
    handles.append(Patch(facecolor="none", edgecolor="none", label="── trips ──"))

    for i, s in enumerate(stats):
        color = cmap(i % 10)
        label = (
            f"T{i+1}: {s['n_customers']:>2}c | "
            f"load {s['demand']:>4.2f} ({s['load_pct']:>3.0f}%) | "
            f"dist {s['distance']:>4.2f}"
        )
        handles.append(
            Line2D(
                [0], [0], color=color, lw=3, marker=">",
                markersize=8, label=label,
            )
        )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
        prop={"family": "monospace"},
    )

    title_parts = []
    if epoch is not None:
        title_parts.append(f"epoch {epoch}")
    title_parts.append(f"total length {-reward:.3f}")
    title_parts.append(f"{len(trips)} trips")
    ax.set_title("  |  ".join(title_parts))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-loc", type=int, default=50)
    parser.add_argument("--variant", default="cvrp")
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Must match the training arch (Tier 2 default 256).",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=30.0,
        help="Seconds between checkpoint polls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the fixed test instance (kept constant across reloads).",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="If set, save one PNG per detected checkpoint update.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable interactive window (use with --save-frames for headless).",
    )
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    patch_env_setstate()

    env = MTVRPEnv(
        generator_params={"num_loc": args.num_loc, "variant_preset": args.variant}
    )

    torch.manual_seed(args.seed)
    td = env.reset(batch_size=[1]).to(device)
    locs = td["locs"][0].cpu().numpy()
    demand = td["demand_linehaul"][0].cpu().numpy()

    if not args.no_display:
        plt.ion()
    fig, ax = plt.subplots(figsize=(13, 8))

    last_mtime = None
    last_reward = None

    print(
        f"Watching {args.checkpoint_dir}/last.ckpt "
        f"(refresh every {args.refresh_interval}s, embed_dim={args.embed_dim})"
    )
    print("Press Ctrl-C to stop.\n")

    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ckpt_path = args.checkpoint_dir / "last.ckpt"

            if not ckpt_path.exists():
                ax.clear()
                ax.text(
                    0.5, 0.5,
                    f"Waiting for {ckpt_path.name}...",
                    ha="center", va="center", fontsize=14,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if args.no_display:
                    time.sleep(args.refresh_interval)
                else:
                    plt.pause(args.refresh_interval)
                continue

            try:
                mtime = ckpt_path.stat().st_mtime
            except FileNotFoundError:
                if args.no_display:
                    time.sleep(args.refresh_interval)
                else:
                    plt.pause(args.refresh_interval)
                continue

            if mtime != last_mtime:
                last_mtime = mtime
                # Brief settle so writer finishes
                time.sleep(1.0)
                try:
                    model, epoch = load_policy(
                        ckpt_path, env, device, args.embed_dim, args.num_heads
                    )
                    actions, reward = get_best_tour(model, env, td)
                    render_tour(fig, ax, td, locs, demand, actions, reward, epoch=epoch)
                    fig.canvas.draw()

                    if args.save_frames:
                        ep = f"{epoch:03d}" if isinstance(epoch, int) else "x"
                        fig.savefig(
                            args.save_frames / f"epoch_{ep}.png",
                            dpi=120,
                            bbox_inches="tight",
                        )

                    delta = (
                        f" (Δ {reward - last_reward:+.3f})"
                        if last_reward is not None
                        else ""
                    )
                    print(
                        f"  epoch {epoch}: reward = {reward:.4f}"
                        f"{delta}  trips={len(decompose_trips(actions))}"
                    )
                    last_reward = reward
                except Exception as e:
                    print(f"  load/render error (will retry): {e}")

            if args.no_display:
                time.sleep(args.refresh_interval)
            else:
                plt.pause(args.refresh_interval)
    except KeyboardInterrupt:
        print("\nViewer stopped.")


if __name__ == "__main__":
    main()
