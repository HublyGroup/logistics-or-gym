import unittest

import numpy as np

from logistics_or_gym.envs.HCVRP import HCVRP


class HCVRPTests(unittest.TestCase):
    def test_can_create_env(self):
        n_vehicles = 3
        n_nodes = 50

        env = HCVRP(n_vehicles=n_vehicles, n_nodes=n_nodes)

        self.assertEqual(env.n_nodes, n_nodes)
        self.assertEqual(env.n_vehicles, n_vehicles)

    def test_cannot_end_with_demand_left(self):
        env = HCVRP(n_nodes=5, n_vehicles=2)

        env.reset()
        env.demand = np.ones(shape=(env.n_nodes,))
        env.node_loc = np.array(
            [[0.0, 0.0], [0.2, 0.2], [0.2, 0.3], [0.4, 0.2], [0.1, 0.6], [1, 1]]
        )

        self.assertFalse(env.is_done())

    def test_is_done_when_no_demand_left(self):
        env = HCVRP(n_nodes=5, n_vehicles=2)

        env.reset()
        env.demand = np.ones(shape=(env.n_nodes,))
        env.node_loc = np.array(
            [[0.0, 0.0], [0.2, 0.2], [0.2, 0.3], [0.4, 0.2], [0.1, 0.6], [1, 1]]
        )

        env.step(selected_node=1, selected_vehicle=0)

        env.step(selected_node=2, selected_vehicle=0)

        _, _, done, _, _ = env.step(selected_node=3, selected_vehicle=0)

        env.step(selected_node=0, selected_vehicle=0)

        env.step(selected_node=4, selected_vehicle=1)

        env.step(selected_node=5, selected_vehicle=1)

        _, _, done, _, _ = env.step(selected_node=0, selected_vehicle=1)

        self.assertTrue(done)
        np.testing.assert_array_equal([0, 1, 2, 3, 0, 0, 0, 0], env.partial_route[0])
        np.testing.assert_array_equal([0, 0, 0, 0, 0, 4, 5, 0], env.partial_route[1])

    def test_works_with_one_vehicle(self):
        env = HCVRP(n_nodes=5, n_vehicles=1)

        env.reset()
        env.demand = np.ones(shape=(env.n_nodes,))
        env.node_loc = np.array(
            [[0.0, 0.0], [0.2, 0.2], [0.2, 0.3], [0.4, 0.2], [0.1, 0.6], [1, 1]]
        )

        env.step(selected_node=1, selected_vehicle=0)

        env.step(selected_node=2, selected_vehicle=0)

        env.step(selected_node=3, selected_vehicle=0)

        env.step(selected_node=4, selected_vehicle=0)

        env.step(selected_node=5, selected_vehicle=0)

        _, _, done, _, _ = env.step(selected_node=0, selected_vehicle=0)

        self.assertTrue(done)
        self.assertEqual(len(env.partial_route), 1)
        np.testing.assert_array_equal(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                0,
            ],
            env.partial_route[0],
        )


if __name__ == "__main__":
    unittest.main()
