from abc import ABC
from typing import Optional, List, Dict

import numpy as np
from gym import spaces, Env


class HCVRP(Env, ABC):
    CAPACITIES = {
        10: [20., 25., 30.],
        20: [20., 25., 30.],
        40: [20., 25., 30.],
        50: [20., 25., 30.],
        60: [20., 25., 30.],
        80: [20., 25., 30.],
        100: [20., 25., 30.],
        120: [20., 25., 30.],
    }

    steps: int = 0
    rewards: List[List[int]] = None
    demand: np.ndarray = None
    node_loc: np.ndarray = None
    partial_route: List[List[int]] = None
    acc_travel_time: np.ndarray = None
    free_capacity: np.ndarray = None
    visited: np.ndarray = None
    prev_vehicle: int = None
    prev_node: int = None

    def __init__(self, n_vehicles=3, n_nodes=50):
        super().__init__()

        self.vehicle_speed = 1
        self.n_nodes = n_nodes
        self.n_vehicles = n_vehicles
        self.action_space = spaces.Box(0, self.n_nodes + self.n_vehicles + 1, shape=(self.n_vehicles, self.n_nodes))
        self.observation_space = spaces.Dict({
            "free_capacity": spaces.Box(0, 100, shape=(n_vehicles, 1)),
            "acc_travel_time": spaces.Box(0, 100, shape=(n_vehicles, 1)),
            "partial_route": spaces.Box(0, n_nodes, shape=(n_vehicles, n_nodes + 1)),
            "node_loc": spaces.Box(0, 1, shape=(n_nodes, 2)),
            "demand": spaces.Box(0, 1, shape=(n_nodes,)),
            "action_mask": spaces.Box(0, 1, shape=(self.n_vehicles, self.n_nodes + 1))
        })

    def render(self, mode="human"):
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        self.free_capacity = np.array([20.0] * self.n_vehicles)
        self.acc_travel_time = np.zeros(shape=(self.n_vehicles,))
        self.partial_route = [[0]] * self.n_vehicles  # starts at depot
        self.node_loc = np.random.uniform(0, 1, size=(self.n_nodes + 1, 2))
        self.demand = np.random.uniform(0, 1, size=(self.n_nodes,))
        self.visited = np.zeros(shape=(self.n_nodes + 1), dtype=bool)

        self.rewards = [[0]] * self.n_vehicles
        self.steps = 0
        self.prev_vehicle: int = -1
        self.prev_node: int = 0

    def _transition(self, current_vehicle: int, selected_vehicle: int, selected_node: int):
        node_loc = self.node_loc[selected_node]
        demand = self.demand[selected_node - 1] if selected_node > 0 else 0

        free_cap: float = self.free_capacity[current_vehicle]
        acc_travel_time: float = self.acc_travel_time[current_vehicle]
        partial_route: List[int] = self.partial_route[current_vehicle]

        new_cap = self._transition_capacity(current_vehicle, selected_vehicle, free_cap, demand)
        new_time, inc_time = self._transition_acc_time(current_vehicle, selected_vehicle, acc_travel_time,
                                                       partial_route,
                                                       node_loc)
        new_path = self._transition_path(current_vehicle, selected_vehicle, partial_route, selected_node)

        if selected_node > 0:
            self.demand[selected_node - 1] = 0

        return new_cap, new_time, new_path, inc_time

    @staticmethod
    def _transition_capacity(current_vehicle: int, selected_vehicle: int, old_cap: float, old_demand: float):
        if current_vehicle == selected_vehicle:
            return old_cap - old_demand

        return old_cap

    def _transition_acc_time(self, current_vehicle: int,
                             selected_vehicle: int,
                             old_time: float,
                             old_path: np.ndarray,
                             node_loc: np.ndarray):
        if current_vehicle == selected_vehicle:
            last_node_loc = old_path[-1]
            dist = np.linalg.norm(last_node_loc - node_loc)
            return old_time + dist / self.vehicle_speed, dist / self.vehicle_speed

        return old_time, 0

    def _transition_path(self, current_vehicle: int,
                         selected_vehicle: int,
                         current_path: np.ndarray,
                         selected_node: int):

        if current_vehicle == selected_vehicle:
            return np.concatenate((current_path, np.array([selected_node])), axis=None)

        return np.concatenate((current_path, np.array([current_path[-1]])), axis=None)

    def reward(self, mode="max") -> float:
        assert mode in ["max", "sum"]
        if mode == "max":
            return - np.max(np.sum(np.array(self.rewards), axis=1))

        elif mode == "sum":
            return - np.sum(np.array(self.rewards))

        else:
            return 0.0

    def get_action_mask(self):

        visited = self.visited[1:]  # Get all nodes except depot.
        free_capacity = self.free_capacity[:, None]
        can_collect = np.repeat(free_capacity, self.n_nodes, axis=1) >= self.demand

        can_collect = can_collect * (visited == 0)

        can_depot = (self.prev_node != 0 or np.alltrue(visited))
        can_depot = np.repeat(np.array([can_depot]), self.n_vehicles)[:, None]

        return np.concatenate([can_depot, can_collect], axis=1)

    def step(self, action: Dict[str, int]):
        selected_node: int = action["node"]
        selected_vehicle: int = action["vehicle"]

        self.visited[selected_node] = True  # or 1

        for vehicle in range(self.n_vehicles):
            new_cap, new_time, new_path, inc_time = self._transition(vehicle, selected_vehicle, selected_node)
            self.rewards[vehicle].append(inc_time)
            self.acc_travel_time[vehicle] = new_time
            self.free_capacity[vehicle] = new_cap
            self.partial_route[vehicle] = new_path

        obs = {
            "free_capacity": self.free_capacity,
            "acc_travel_time": self.acc_travel_time,
            "partial_route": self.partial_route,
            "node_loc": self.node_loc,
            "demand": self.demand,
            "action_mask": self.get_action_mask()
        }

        cost = self.reward()
        partial_route = np.array(self.partial_route)
        is_done = np.alltrue(self.node_loc[0] == self.node_loc[partial_route[:, -1]]) and np.alltrue(self.visited)
        self.steps += 1

        return obs, cost, is_done, {}
