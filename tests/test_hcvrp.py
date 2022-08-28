import unittest

import numpy as np

from routing.envs.HCVRP import HCVRP


class HCVRPTests(unittest.TestCase):
    def test_can_create_env(self):
        n_vehicles = 3
        n_nodes = 50

        env = HCVRP(n_vehicles=n_vehicles, n_nodes=n_nodes)

        self.assertEqual(env.n_nodes, n_nodes)
        self.assertEqual(env.n_vehicles, n_vehicles)

    def test_can_call_reset(self):
        env = HCVRP(n_nodes=5, n_vehicles=2)

        env.reset()
        env.demand = np.ones(shape=(env.n_nodes,))
        env.node_loc = np.array([[0.0, 0.0], [0.2, 0.2], [0.2, 0.3], [0.4, 0.2], [0.1, 0.6], [1, 1]])

        action = {"node": 1, "vehicle": 0}
        env.step(action)

        action = {"node": 2, "vehicle": 0}
        env.step(action)

        action = {"node": 3, "vehicle": 0}
        env.step(action)

        action = {"node": 0, "vehicle": 0}
        env.step(action)

        action = {"node": 4, "vehicle": 1}
        env.step(action)

        action = {"node": 5, "vehicle": 1}
        env.step(action)

        action = {"node": 0, "vehicle": 1}
        _, _, done, _ = env.step(action)

        self.assertTrue(done)


if __name__ == '__main__':
    unittest.main()
