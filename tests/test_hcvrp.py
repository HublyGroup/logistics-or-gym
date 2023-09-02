import unittest

import numpy as np

from logistics_or_gym.envs.HeterogeneousCVRP import HeterogeneousCVRP


class HCVRPTests(unittest.TestCase):
    def test_can_create_env(self):
        n_vehicles = 3
        n_nodes = 50

        env = HeterogeneousCVRP(n_vehicles=n_vehicles, n_nodes=n_nodes)

        self.assertEqual(env.n_nodes, n_nodes)
        self.assertEqual(env.n_vehicles, n_vehicles)

    def test_cannot_end_with_demand_left(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)

        env.reset()
        env.node_loc = np.array(
            [[0.0, 0.0], [0.2, 0.2], [0.2, 0.3], [0.4, 0.2], [0.1, 0.6], [1, 1]]
        )

        self.assertFalse(env.is_done())

    def test_is_done_when_no_demand_left(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)

        env.reset()

        env.step(np.array([0, 1]))

        env.step(np.array([0, 2]))

        _, _, done, _, _ = env.step(np.array([0, 3]))

        env.step(np.array([0, 0]))

        env.step(np.array([1, 4]))

        env.step(np.array([1, 5]))

        _, _, done, _, _ = env.step(np.array([1, 0]))

        self.assertTrue(done)
        np.testing.assert_array_equal([0, 1, 2, 3, 0, 0, 0, 0], env.partial_routes[0])
        np.testing.assert_array_equal([0, 0, 0, 0, 0, 4, 5, 0], env.partial_routes[1])

    def test_works_with_one_vehicle(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=1)

        env.reset()

        env.step(np.array([0, 1]))

        env.step(np.array([0, 2]))

        env.step(np.array([0, 3]))

        env.step(np.array([0, 4]))

        env.step(np.array([0, 5]))

        _, _, done, _, _ = env.step(np.array([0, 0]))

        self.assertTrue(done)
        self.assertEqual(len(env.partial_routes), 1)
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
            env.partial_routes[0],
        )


if __name__ == "__main__":
    unittest.main()
