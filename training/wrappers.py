"""Gym wrappers that adapt HeterogeneousCVRP for off-the-shelf RL libraries.

The env's native action space is `MultiDiscrete([n_vehicles, n_total_nodes])` and
its native action mask is a 2D `(n_vehicles, n_total_nodes)` joint mask. SB3's
MaskablePPO expects a flat `Discrete(n)` space with a flat boolean mask, so we
flatten both — at the cost of dropping the joint structure (each MultiDiscrete
dim is masked independently in MaskablePPO's MultiDiscrete support, which loses
the vehicle×node correlations our env encodes).

We also drop `partial_routes` from the observation — it's a `Sequence` space,
which SB3's MultiInputPolicy can't tensorize. Agents still see capacity, time,
locations, and demand, which is enough information to act.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class FlatMaskableWrapper(gym.Wrapper):
    """Flatten MultiDiscrete([V, N]) → Discrete(V*N) and expose `action_masks()`.

    Action encoding: `flat = vehicle * n_total_nodes + node`.
    Decoding: `vehicle, node = divmod(flat, n_total_nodes)`.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        nvec = env.action_space.nvec
        self.n_vehicles = int(nvec[0])
        self.n_total_nodes = int(nvec[1])
        self.action_space = gym.spaces.Discrete(
            self.n_vehicles * self.n_total_nodes
        )

        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces.pop("partial_routes", None)
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._strip_obs(obs), info

    def step(self, action):
        vehicle, node = divmod(int(action), self.n_total_nodes)
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([vehicle, node])
        )
        return self._strip_obs(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = self.env.unwrapped.get_action_mask()
        return mask.reshape(-1).astype(bool)

    @staticmethod
    def _strip_obs(obs: dict) -> dict:
        return {k: v for k, v in obs.items() if k != "partial_routes"}
