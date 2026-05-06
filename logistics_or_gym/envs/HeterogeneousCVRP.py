from abc import ABC
from typing import Literal, Optional

import numpy as np
from gymnasium import Env, spaces
from gymnasium.spaces import MultiDiscrete

Objective = Literal["min_max", "min_sum", "min_max_cost", "min_sum_cost"]
DemandDist = Literal["uniform", "discrete"]


class HeterogeneousCVRP(Env, ABC):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    steps: int = 0
    demand: np.ndarray = None
    node_loc: np.ndarray = None
    partial_routes: list[list[int]] = None
    acc_travel_time: np.ndarray = None
    acc_cost: np.ndarray = None
    free_capacity: np.ndarray = None
    visited: np.ndarray = None
    deployed: np.ndarray = None

    def __init__(
        self,
        n_vehicles: int = 2,
        n_nodes: int = 50,
        n_depots: int = 1,
        capacities: Optional[list[float]] = None,
        speeds: Optional[list[float]] = None,
        fixed_costs: Optional[list[float]] = None,
        variable_costs: Optional[list[float]] = None,
        objective: Objective = "min_max",
        demand_dist: DemandDist = "uniform",
        render_mode: Optional[str] = None,
    ):
        """
        :param n_vehicles: number of vehicles in the fleet.
        :param n_nodes: number of customer nodes (excluding depots).
        :param n_depots: number of depots.
        :param capacities: per-vehicle capacity. Defaults to 1.0 for each vehicle
            (paired with `demand_dist="uniform"`). For paper-style instances pass
            integer capacities like [20, 25, 30] together with `demand_dist="discrete"`.
            Currently scalar per vehicle; the trailing dim is reserved for
            multi-dimensional capacities (weight + volume + ...) — see issue tracker.
        :param speeds: per-vehicle speed. Defaults to 1.0 for each vehicle. For
            MS-HCVRP the paper uses speeds inversely proportional to capacity, e.g.
            [1/4, 1/5, 1/6] for V3.
        :param fixed_costs: per-vehicle deployment cost, charged once when the
            vehicle is first selected to move. Defaults to 0.0 (no fixed cost,
            current behaviour). Models real fleet economics (paper 1's `fc_k`).
        :param variable_costs: per-vehicle cost per unit distance. Defaults to
            1.0 (cost == distance, current behaviour). Decouples cost from speed
            so a fast vehicle can also be expensive per km (paper 1's `ac_k`).
        :param objective: "min_max" / "min_sum" → reward based on travel time
            (makespan / total time, the original Li et al. setup). "min_max_cost"
            / "min_sum_cost" → reward based on total cost
            (`fc * deployed + vc * distance`, paper 1's MILP objective).
        :param demand_dist: "uniform" → demands ~ U(0, 1); "discrete" → demands ~
            uniform integer in {1,...,9}, matching the paper's data generator.
        :param render_mode: "human" (live matplotlib window) or "rgb_array"
            (returns an HxWx3 uint8 image). None disables rendering. Requires
            matplotlib installed (e.g. `pip install logistics-or-gym[viz]`).
        """
        super().__init__()

        assert objective in ("min_max", "min_sum", "min_max_cost", "min_sum_cost")
        assert demand_dist in ("uniform", "discrete")
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._fig = None
        self._ax = None

        self.n_depots = n_depots
        self.n_nodes = n_nodes
        self.n_vehicles = n_vehicles
        self.objective: Objective = objective
        self.demand_dist: DemandDist = demand_dist

        self.action_space = MultiDiscrete(
            [self.n_vehicles, self.n_nodes + self.n_depots]
        )

        if capacities is None:
            self.capacities = np.ones((self.n_vehicles, 1), dtype=np.float32)
        else:
            assert len(capacities) == self.n_vehicles
            self.capacities = np.array(capacities, dtype=np.float32).reshape(
                self.n_vehicles, 1
            )

        if speeds is None:
            self.speeds = np.ones((self.n_vehicles, 1), dtype=np.float32)
        else:
            assert len(speeds) == self.n_vehicles
            self.speeds = np.array(speeds, dtype=np.float32).reshape(
                self.n_vehicles, 1
            )

        if fixed_costs is None:
            self.fixed_costs = np.zeros((self.n_vehicles, 1), dtype=np.float32)
        else:
            assert len(fixed_costs) == self.n_vehicles
            self.fixed_costs = np.array(fixed_costs, dtype=np.float32).reshape(
                self.n_vehicles, 1
            )

        if variable_costs is None:
            self.variable_costs = np.ones((self.n_vehicles, 1), dtype=np.float32)
        else:
            assert len(variable_costs) == self.n_vehicles
            self.variable_costs = np.array(
                variable_costs, dtype=np.float32
            ).reshape(self.n_vehicles, 1)

        demand_high = 1.0 if demand_dist == "uniform" else 9.0
        cap_high = float(self.capacities.max())

        self.max_step = 1000
        self.observation_space = spaces.Dict(
            {
                "free_capacity": spaces.Box(
                    0, cap_high, shape=(self.n_vehicles, 1), dtype=np.float32
                ),
                "acc_travel_time": spaces.Box(
                    0, np.inf, shape=(self.n_vehicles, 1), dtype=np.float32
                ),
                "partial_routes": spaces.Sequence(
                    spaces.Sequence(spaces.Discrete(self.n_depots + self.n_nodes))
                ),
                "depots_idx": spaces.Box(
                    0, self.n_depots, shape=(self.n_depots,), dtype=np.int32
                ),
                "demands_idx": spaces.Box(
                    self.n_depots,
                    self.n_depots + self.n_nodes,
                    shape=(self.n_nodes,),
                    dtype=np.int32,
                ),
                "node_loc": spaces.Box(
                    0, 1, shape=(self.n_depots + self.n_nodes, 2), dtype=np.float32
                ),
                "demand": spaces.Box(
                    0,
                    demand_high,
                    shape=(self.n_depots + self.n_nodes, 1),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    0,
                    1,
                    shape=(self.n_vehicles, self.n_depots + self.n_nodes),
                    dtype=np.int8,
                ),
                "fixed_costs": spaces.Box(
                    0, np.inf, shape=(self.n_vehicles, 1), dtype=np.float32
                ),
                "variable_costs": spaces.Box(
                    0, np.inf, shape=(self.n_vehicles, 1), dtype=np.float32
                ),
                "acc_cost": spaces.Box(
                    0, np.inf, shape=(self.n_vehicles, 1), dtype=np.float32
                ),
                "deployed": spaces.Box(
                    0, 1, shape=(self.n_vehicles,), dtype=np.int8
                ),
            }
        )

    def render(self):
        if self.render_mode is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "render() requires matplotlib. Install it via "
                "`pip install logistics-or-gym[viz]` or `pip install matplotlib`."
            ) from e

        if self.node_loc is None:
            raise RuntimeError("Call reset() before render().")

        if self._fig is None:
            if self.render_mode == "human":
                plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 7))

        ax = self._ax
        ax.clear()

        depot_locs = self.node_loc[: self.n_depots]
        ax.scatter(
            depot_locs[:, 0],
            depot_locs[:, 1],
            marker="s",
            s=240,
            c="red",
            edgecolors="black",
            label="depot",
            zorder=5,
        )

        demand_locs = self.node_loc[self.n_depots :]
        demand_vals = self.demand[self.n_depots :].flatten()
        visited_demand = self.visited[self.n_depots :]
        max_d = max(float(demand_vals.max()), 1e-9)
        sizes = 20.0 + 80.0 * (demand_vals / max_d)

        if (~visited_demand).any():
            ax.scatter(
                demand_locs[~visited_demand, 0],
                demand_locs[~visited_demand, 1],
                s=sizes[~visited_demand],
                c="lightgray",
                edgecolors="black",
                label="unvisited",
                zorder=3,
            )
        if visited_demand.any():
            ax.scatter(
                demand_locs[visited_demand, 0],
                demand_locs[visited_demand, 1],
                s=sizes[visited_demand],
                c="black",
                alpha=0.35,
                edgecolors="black",
                label="visited",
                zorder=3,
            )

        cmap = plt.get_cmap("tab10")
        for v_idx, route in enumerate(self.partial_routes):
            color = cmap(v_idx % 10)
            pts = self.node_loc[np.array(route)]

            if len(route) >= 2:
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    "-",
                    color=color,
                    linewidth=2.0,
                    alpha=0.7,
                    zorder=4,
                )

            cx, cy = float(pts[-1, 0]), float(pts[-1, 1])
            ax.scatter(
                [cx],
                [cy],
                marker="^",
                s=260,
                facecolors=[color],
                edgecolors="black",
                linewidths=1.5,
                zorder=7,
                label=f"vehicle {v_idx}",
            )
            ax.annotate(
                f"V{v_idx}",
                (cx, cy),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=color,
                zorder=8,
            )

        makespan = float(np.max(self.acc_travel_time))
        total_time = float(np.sum(self.acc_travel_time))
        max_cost = float(np.max(self.acc_cost))
        total_cost = float(np.sum(self.acc_cost))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(
            f"step {self.steps}  |  makespan {makespan:.3f}  |  "
            f"Σtime {total_time:.3f}  |  Σcost {total_cost:.3f}  |  "
            f"max-cost {max_cost:.3f}"
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.3)

        self._fig.canvas.draw()

        if self.render_mode == "human":
            self._fig.canvas.flush_events()
            plt.pause(1.0 / self.metadata["render_fps"])
            return None

        img = np.asarray(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        return img[:, :, :3].copy()

    def close(self):
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except ImportError:
                pass
            self._fig = None
            self._ax = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.free_capacity = self.capacities.copy()
        self.acc_travel_time = np.zeros(
            shape=(self.n_vehicles, 1), dtype=np.float32
        )
        self.acc_cost = np.zeros(
            shape=(self.n_vehicles, 1), dtype=np.float32
        )
        self.deployed = np.zeros(self.n_vehicles, dtype=bool)
        self.partial_routes = [
            [int(self.np_random.integers(0, self.n_depots))]
            for _ in range(self.n_vehicles)
        ]
        self.node_loc = self.np_random.uniform(
            0, 1, size=(self.n_depots + self.n_nodes, 2)
        ).astype(np.float32)

        if self.demand_dist == "uniform":
            self.demand = self.np_random.uniform(
                0, 1, size=(self.n_depots + self.n_nodes, 1)
            ).astype(np.float32)
        else:
            self.demand = self.np_random.integers(
                1, 10, size=(self.n_depots + self.n_nodes, 1)
            ).astype(np.float32)
        self.demand[: self.n_depots] = 0

        self.visited = np.zeros(
            shape=(self.n_depots + self.n_nodes), dtype=bool
        )
        # Depots have no demand, so they're "served" from the start. Marking all
        # of them visited keeps `is_done` correct when n_depots > n_vehicles
        # (otherwise unstarted depots would block termination).
        self.visited[: self.n_depots] = True
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "free_capacity": self.free_capacity,
            "acc_travel_time": self.acc_travel_time,
            "partial_routes": tuple(tuple(r) for r in self.partial_routes),
            "node_loc": self.node_loc,
            "demand": self.demand,
            "depots_idx": np.array(list(range(self.n_depots)), dtype=np.int32),
            "demands_idx": np.array(
                list(range(self.n_depots, self.n_depots + self.n_nodes)),
                dtype=np.int32,
            ),
            "action_mask": self.get_action_mask(),
            "fixed_costs": self.fixed_costs,
            "variable_costs": self.variable_costs,
            "acc_cost": self.acc_cost,
            "deployed": self.deployed.astype(np.int8),
        }

    def _transition(
        self, current_vehicle: int, selected_vehicle: int, selected_node: int
    ):
        """The Transition function is a direct implementation of the transition function from:
        http://arxiv.org/abs/2110.02629 http://dx.doi.org/10.1109/TCYB.2021.3111082
        """
        node_loc = self.node_loc[selected_node]
        demand = self.demand[selected_node]

        free_cap: float = self.free_capacity[current_vehicle]
        acc_travel_time: float = self.acc_travel_time[current_vehicle]
        partial_route: list[int] = self.partial_routes[current_vehicle]

        new_cap = self._transition_capacity(
            current_vehicle, selected_vehicle, free_cap, demand
        )
        new_time, inc_time = self._transition_acc_time(
            current_vehicle, selected_vehicle, acc_travel_time, partial_route, node_loc
        )
        new_path = self._transition_path(
            current_vehicle, selected_vehicle, partial_route, selected_node
        )

        return new_cap, new_time, new_path, inc_time

    @staticmethod
    def _transition_capacity(
        current_vehicle: int, selected_vehicle: int, old_cap: float, old_demand: float
    ):
        if current_vehicle == selected_vehicle:
            return old_cap - old_demand

        return old_cap

    def _transition_acc_time(
        self,
        current_vehicle: int,
        selected_vehicle: int,
        old_time: float,
        old_path: list[int],
        node_loc: np.ndarray,
    ):
        if current_vehicle == selected_vehicle:
            last_node_loc = self.node_loc[old_path[-1]]
            dist = float(np.linalg.norm(last_node_loc - node_loc))
            speed = float(self.speeds[current_vehicle, 0])
            inc = dist / speed
            return old_time + inc, inc

        return old_time, 0.0

    def _transition_path(
        self,
        current_vehicle: int,
        selected_vehicle: int,
        current_path: list[int],
        selected_node: int,
    ):
        if current_vehicle == selected_vehicle:
            current_path.append(int(selected_node))
            return current_path

        current_path.append(current_path[-1])
        return current_path

    def reward(self) -> float:
        """Cumulative cost so far, returned as a negative number.

        Time-based: ``min_max`` (-makespan) / ``min_sum`` (-total time).
        Cost-based: ``min_max_cost`` (-max per-vehicle cost) /
        ``min_sum_cost`` (-total fleet cost = Σ fc·deployed + vc·distance).
        """
        if self.objective == "min_max":
            return -float(np.max(self.acc_travel_time))
        if self.objective == "min_sum":
            return -float(np.sum(self.acc_travel_time))
        if self.objective == "min_max_cost":
            return -float(np.max(self.acc_cost))
        return -float(np.sum(self.acc_cost))

    def get_action_mask(self):
        visited_demand = self.visited[self.n_depots :]  # (n_nodes,) bool
        free_cap = self.free_capacity.reshape(self.n_vehicles)
        demand_per_node = self.demand[self.n_depots :].reshape(self.n_nodes)

        can_collect = (free_cap[:, None] >= demand_per_node[None, :]) & (
            ~visited_demand
        )[None, :]

        prev_nodes = np.array([route[-1] for route in self.partial_routes])
        not_at_depot = prev_nodes >= self.n_depots
        all_visited = bool(np.all(self.visited))
        can_depot_per_vehicle = not_at_depot | all_visited
        can_depot = np.broadcast_to(
            can_depot_per_vehicle[:, None], (self.n_vehicles, self.n_depots)
        )

        return np.concatenate([can_depot, can_collect], axis=1).astype(np.int8)

    def is_done(self) -> bool:
        last_nodes = np.array([route[-1] for route in self.partial_routes])
        all_at_depot = bool(np.all(last_nodes < self.n_depots))
        return all_at_depot and bool(np.all(self.visited))

    def step(self, actions: np.ndarray):
        selected_vehicle = int(actions[0])
        selected_node = int(actions[1])
        assert selected_vehicle < self.n_vehicles

        self.visited[selected_node] = True

        inc_time_selected = 0.0
        for vehicle in range(self.n_vehicles):
            new_cap, new_time, new_path, inc_time = self._transition(
                vehicle, selected_vehicle, selected_node
            )
            self.acc_travel_time[vehicle] = new_time
            self.free_capacity[vehicle] = new_cap
            self.partial_routes[vehicle] = new_path
            if vehicle == selected_vehicle:
                inc_time_selected = float(inc_time)

        # Cost accounting for the selected vehicle this step.
        # inc_distance = inc_time * speed (since inc_time = dist / speed).
        speed = float(self.speeds[selected_vehicle, 0])
        inc_distance_selected = inc_time_selected * speed
        vc = float(self.variable_costs[selected_vehicle, 0])
        fc = (
            float(self.fixed_costs[selected_vehicle, 0])
            if (inc_time_selected > 0 and not self.deployed[selected_vehicle])
            else 0.0
        )
        inc_cost_selected = fc + vc * inc_distance_selected
        self.acc_cost[selected_vehicle, 0] += inc_cost_selected
        if inc_time_selected > 0:
            self.deployed[selected_vehicle] = True

        if selected_node < self.n_depots:
            self.free_capacity[selected_vehicle] = self.capacities[selected_vehicle]
        else:
            self.demand[selected_node] = 0

        is_done = self.is_done()
        self.steps += 1
        truncated = self.steps >= self.max_step
        episode_end = is_done or truncated

        if self.objective == "min_sum":
            step_reward = -inc_time_selected
        elif self.objective == "min_sum_cost":
            step_reward = -inc_cost_selected
        elif self.objective == "min_max":
            step_reward = (
                -float(np.max(self.acc_travel_time)) if episode_end else 0.0
            )
        else:  # min_max_cost
            step_reward = (
                -float(np.max(self.acc_cost)) if episode_end else 0.0
            )

        return self._get_obs(), step_reward, is_done, truncated, {}
