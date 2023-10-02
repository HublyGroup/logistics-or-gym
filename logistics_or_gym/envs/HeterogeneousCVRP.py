from abc import ABC
from typing import Optional

import numpy as np
from gymnasium import spaces, Env
from gymnasium.spaces import MultiDiscrete


class HeterogeneousCVRP(Env, ABC):
    steps: int = 0
    rewards: list[list[int]] = None
    demand: np.ndarray = None
    node_loc: np.ndarray = None
    partial_routes: list[list[int]] = None
    acc_travel_time: np.ndarray = None
    free_capacity: np.ndarray = None
    visited: np.ndarray = None
    prev_vehicle: int = None
    prev_node: int = None

    def __init__(
        self,
        n_vehicles: int = 2,
        n_nodes: int = 50,
        n_depots: int = 1,
        capacities: Optional[list[float]] = None,
    ):
        """

        :param n_vehicles (int): The number of vehicles in the plan. Defaults to 2
        :param n_nodes (int): number of nodes in the graph aka. the locations where demands needs to be picked up.
        Defaults to 50
        :param n_depots (int): The number of depots. Defaults to 1.
        :param capacities (list[float]): Specify the capacity of each vehicle. Defaults to None which means a normalized
        capacity of 1 will be applied to all vehicles
        """
        super().__init__()

        self.n_depots = n_depots
        self.vehicle_speed = 1
        self.n_nodes = n_nodes
        self.n_vehicles = n_vehicles
        self.action_space = MultiDiscrete(
            [self.n_vehicles, self.n_nodes + self.n_depots]
        )

        if capacities is None:
            self.capacities = [1.0] * n_vehicles
        else:
            assert len(capacities) == self.n_vehicles

        self.max_step = 1000
        self.observation_space = spaces.Dict(
            {
                "free_capacity": spaces.Box(0, 100, shape=(self.n_vehicles, 1)),
                "acc_travel_time": spaces.Box(0, 100, shape=(self.n_vehicles, 1)),
                "partial_routes": spaces.Sequence(
                    spaces.Sequence(spaces.Discrete(self.n_depots + self.n_nodes))
                ),
                "node_loc": spaces.Box(0, 1, shape=(self.n_depots + self.n_nodes, 2)),
                "demand": spaces.Box(0, 1, shape=(self.n_nodes, 1)),
                "action_mask": spaces.Box(
                    0, 1, shape=(self.n_vehicles, self.n_depots + self.n_nodes)
                ),
            }
        )

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            np.random.seed(seed=seed)

        self.free_capacity = np.array(self.capacities)
        self.acc_travel_time = np.zeros(shape=(self.n_vehicles, 1))
        self.partial_routes = [
            [np.random.randint(0, self.n_depots)] for _ in range(self.n_vehicles)
        ]  # starts at depot
        self.node_loc = np.random.uniform(0, 1, size=(self.n_depots + self.n_nodes, 2))
        self.demand = np.random.uniform(
            0, 1, size=(self.n_depots + self.n_nodes, 1)
        )  # Include depots so the indexes matches
        self.demand[list(range(self.n_depots))] = 0
        self.visited = np.zeros(
            shape=(self.n_depots + self.n_nodes), dtype=bool
        )  # Include depots so the indexes matches
        self.visited[list(range(self.n_depots))] = [
            d in [pr[0] for pr in self.partial_routes] for d in range(self.n_depots)
        ]
        self.rewards = [[0]] * self.n_vehicles
        self.steps = 0
        self.prev_vehicle: int = -1
        self.prev_node: int = 0

        obs = {
            "free_capacity": self.free_capacity,
            "acc_travel_time": self.acc_travel_time,
            "partial_routes": self.partial_routes,
            "node_loc": self.node_loc,
            "demand": self.demand,
            "action_mask": self.get_action_mask(),
        }

        return obs, {}

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

        if selected_node > self.n_depots:
            self.demand[selected_node] = 0

        if self.n_depots < selected_node:
            self.free_capacity[selected_vehicle] = 0

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
            dist = np.linalg.norm(last_node_loc - node_loc)
            return old_time + dist / self.vehicle_speed, dist / self.vehicle_speed

        return old_time, 0

    def _transition_path(
        self,
        current_vehicle: int,
        selected_vehicle: int,
        current_path: list[int],
        selected_node: int,
    ):
        if current_vehicle == selected_vehicle:
            current_path.append(selected_node)
            return current_path

        current_path.append(current_path[-1])
        return current_path

    def reward(self, mode="max") -> float:
        assert mode in ["max", "sum"]
        if mode == "max":
            return -np.max(np.sum(np.array(self.rewards), axis=1))

        elif mode == "sum":
            return -np.sum(np.array(self.rewards))

        else:
            return 0.0

    def get_action_mask(self):
        visited = self.visited[self.n_depots :]  # Get all nodes except depot.
        free_capacity = self.free_capacity[:, None]
        can_collect = (
            np.repeat(free_capacity, self.n_nodes, axis=1)
            >= self.demand[self.n_depots :]
        )

        can_collect = can_collect * (visited == 0)

        can_depot = self.prev_node != 0 or np.alltrue(visited)
        can_depot = np.repeat(np.array([can_depot]), self.n_vehicles)[:, None]

        return np.concatenate([can_depot, can_collect], axis=1)

    def is_done(self) -> bool:
        partial_route = np.array(self.partial_routes)
        return np.alltrue(
            self.node_loc[0] == self.node_loc[partial_route[:, -1]]
        ) and np.alltrue(self.visited)

    def step(self, actions: np.ndarray):
        selected_vehicle = actions[0]
        selected_node = actions[1]
        assert selected_vehicle < self.n_vehicles

        self.visited[selected_node] = True  # or 1

        for vehicle in range(self.n_vehicles):
            new_cap, new_time, new_path, inc_time = self._transition(
                vehicle, selected_vehicle, selected_node
            )
            self.rewards[vehicle].append(inc_time)
            self.acc_travel_time[vehicle] = new_time
            self.free_capacity[vehicle] = new_cap
            self.partial_routes[vehicle] = new_path

        obs = {
            "free_capacity": self.free_capacity,
            "acc_travel_time": self.acc_travel_time,
            "partial_route": self.partial_routes,
            "node_loc": self.node_loc,
            "demand": self.demand,
            "action_mask": self.get_action_mask(),
        }

        cost = self.reward()

        is_done = self.is_done()
        self.steps += 1

        truncated = self.steps >= self.max_step

        return obs, cost, is_done, truncated, {}
