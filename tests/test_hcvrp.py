import unittest

import matplotlib

matplotlib.use("Agg")

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


class HCVRPTransitionTests(unittest.TestCase):
    def test_capacity_refills_on_depot_return(self):
        env = HeterogeneousCVRP(n_nodes=3, n_vehicles=1, capacities=[1.0])
        env.reset(seed=0)

        env.step(np.array([0, 1]))
        cap_after_pickup = float(env.free_capacity[0, 0])
        self.assertLess(cap_after_pickup, 1.0)

        env.step(np.array([0, 0]))
        self.assertEqual(float(env.free_capacity[0, 0]), 1.0)

    def test_custom_capacities_refill_to_their_own_value(self):
        env = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=2, capacities=[5.0, 10.0]
        )
        env.reset(seed=0)
        np.testing.assert_array_equal(env.free_capacity.flatten(), [5.0, 10.0])

        env.step(np.array([0, 1]))
        self.assertLess(float(env.free_capacity[0, 0]), 5.0)

        env.step(np.array([0, 0]))
        self.assertEqual(float(env.free_capacity[0, 0]), 5.0)

    def test_demand_zeroed_on_pickup(self):
        env = HeterogeneousCVRP(n_nodes=3, n_vehicles=1)
        env.reset(seed=0)
        self.assertGreater(float(env.demand[1, 0]), 0)

        env.step(np.array([0, 1]))
        self.assertEqual(float(env.demand[1, 0]), 0.0)

    def test_unselected_vehicles_repeat_last_node(self):
        env = HeterogeneousCVRP(n_nodes=4, n_vehicles=3)
        env.reset(seed=0)

        env.step(np.array([1, 2]))

        self.assertEqual(env.partial_routes[0], [0, 0])
        self.assertEqual(env.partial_routes[1], [0, 2])
        self.assertEqual(env.partial_routes[2], [0, 0])

    def test_partial_routes_stay_same_length(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=3)
        env.reset(seed=0)
        for action in [(0, 1), (1, 2), (2, 3), (0, 0), (1, 0)]:
            env.step(np.array(action))

        lengths = {len(r) for r in env.partial_routes}
        self.assertEqual(len(lengths), 1)

    def test_per_vehicle_speed_scales_travel_time(self):
        env = HeterogeneousCVRP(
            n_nodes=2,
            n_vehicles=2,
            capacities=[10.0, 10.0],
            speeds=[1.0, 0.5],
            demand_dist="discrete",
        )

        env.reset(seed=0)
        env.node_loc = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        env.step(np.array([0, 1]))
        self.assertAlmostEqual(float(env.acc_travel_time[0, 0]), 1.0, places=5)

        env.reset(seed=0)
        env.node_loc = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        env.step(np.array([1, 1]))
        self.assertAlmostEqual(float(env.acc_travel_time[1, 0]), 2.0, places=5)


class HCVRPRewardTests(unittest.TestCase):
    ROUTE = [(0, 1), (0, 2), (0, 3), (0, 0), (1, 4), (1, 5), (1, 0)]

    def test_min_sum_step_rewards_sum_to_negative_total_time(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2, objective="min_sum"
        )
        env.reset(seed=7)

        total_reward = 0.0
        for action in self.ROUTE:
            _, r, *_ = env.step(np.array(action))
            total_reward += r

        expected = -float(env.acc_travel_time.sum())
        self.assertAlmostEqual(total_reward, expected, places=5)

    def test_min_max_rewards_zero_until_terminal(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2, objective="min_max"
        )
        env.reset(seed=7)

        rewards = []
        done = False
        for action in self.ROUTE:
            _, r, done, _, _ = env.step(np.array(action))
            rewards.append(r)

        self.assertTrue(done)
        for r in rewards[:-1]:
            self.assertEqual(r, 0.0)
        self.assertLess(rewards[-1], 0.0)
        self.assertAlmostEqual(
            rewards[-1], -float(np.max(env.acc_travel_time)), places=5
        )

    def test_reward_method_matches_objective(self):
        env_max = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=2, objective="min_max"
        )
        env_max.reset(seed=0)
        env_max.acc_travel_time = np.array([[2.0], [5.0]], dtype=np.float32)
        self.assertAlmostEqual(env_max.reward(), -5.0, places=5)

        env_sum = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=2, objective="min_sum"
        )
        env_sum.reset(seed=0)
        env_sum.acc_travel_time = np.array([[2.0], [5.0]], dtype=np.float32)
        self.assertAlmostEqual(env_sum.reward(), -7.0, places=5)


class HCVRPMaskTests(unittest.TestCase):
    def test_initial_mask_blocks_depot_and_allows_all_demand(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)
        obs, _ = env.reset(seed=0)

        mask = obs["action_mask"]
        self.assertEqual(mask.shape, (2, 6))
        self.assertEqual(mask.dtype, np.int8)
        np.testing.assert_array_equal(mask[:, 0], [0, 0])
        self.assertTrue((mask[:, 1:] == 1).all())

    def test_mask_blocks_visited_demand_node(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)
        env.reset(seed=0)
        obs, *_ = env.step(np.array([0, 1]))

        mask = obs["action_mask"]
        np.testing.assert_array_equal(mask[:, 1], [0, 0])

    def test_mask_blocks_overcapacity_demand(self):
        env = HeterogeneousCVRP(n_nodes=2, n_vehicles=1, capacities=[1.0])
        env.reset(seed=0)
        env.demand = np.array([[0.0], [0.5], [2.0]], dtype=np.float32)
        env.visited = np.array([True, False, False])

        mask = env.get_action_mask()
        self.assertEqual(mask[0, 1], 1)
        self.assertEqual(mask[0, 2], 0)

    def test_mask_allows_depot_when_all_demand_visited(self):
        env = HeterogeneousCVRP(n_nodes=2, n_vehicles=1)
        env.reset(seed=0)
        env.visited = np.array([True, True, True])

        mask = env.get_action_mask()
        self.assertEqual(mask[0, 0], 1)

    def test_multi_depot_mask_opens_all_depot_cols_for_moving_vehicle(self):
        env = HeterogeneousCVRP(n_nodes=4, n_vehicles=2, n_depots=3)
        env.reset(seed=0)

        first_demand_node = env.n_depots
        obs, *_ = env.step(np.array([0, first_demand_node]))
        mask = obs["action_mask"]

        np.testing.assert_array_equal(mask[0, : env.n_depots], [1, 1, 1])


class HCVRPConfigTests(unittest.TestCase):
    def test_discrete_demands_are_integers_in_one_to_nine(self):
        env = HeterogeneousCVRP(
            n_nodes=20,
            n_vehicles=2,
            capacities=[20.0, 25.0],
            demand_dist="discrete",
        )
        env.reset(seed=42)

        self.assertEqual(float(env.demand[0, 0]), 0.0)
        customer_demands = env.demand[1:].flatten()
        for d in customer_demands:
            self.assertEqual(d, int(d))
            self.assertGreaterEqual(int(d), 1)
            self.assertLessEqual(int(d), 9)

    def test_uniform_demands_are_in_zero_one(self):
        env = HeterogeneousCVRP(n_nodes=20, n_vehicles=2)
        env.reset(seed=42)

        self.assertEqual(float(env.demand[0, 0]), 0.0)
        customer_demands = env.demand[1:].flatten()
        self.assertTrue((customer_demands > 0).all())
        self.assertTrue((customer_demands <= 1).all())

    def test_multi_depot_observation_shapes(self):
        env = HeterogeneousCVRP(n_nodes=4, n_vehicles=2, n_depots=3)
        obs, _ = env.reset(seed=0)

        self.assertEqual(obs["action_mask"].shape, (2, 7))
        self.assertEqual(obs["node_loc"].shape, (7, 2))
        self.assertEqual(obs["demand"].shape, (7, 1))
        np.testing.assert_array_equal(obs["depots_idx"], [0, 1, 2])
        np.testing.assert_array_equal(obs["demands_idx"], [3, 4, 5, 6])

    def test_seed_makes_reset_reproducible(self):
        env_a = HeterogeneousCVRP(n_nodes=10, n_vehicles=2)
        obs_a, _ = env_a.reset(seed=123)

        env_b = HeterogeneousCVRP(n_nodes=10, n_vehicles=2)
        obs_b, _ = env_b.reset(seed=123)

        np.testing.assert_array_equal(obs_a["node_loc"], obs_b["node_loc"])
        np.testing.assert_array_equal(obs_a["demand"], obs_b["demand"])

    def test_truncation_at_max_step(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=1)
        env.reset(seed=0)
        env.max_step = 3

        truncated = False
        for _ in range(3):
            _, _, _, truncated, _ = env.step(np.array([0, 0]))

        self.assertTrue(truncated)

    def test_reset_and_step_obs_have_same_keys(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)
        reset_obs, _ = env.reset(seed=0)
        step_obs, *_ = env.step(np.array([0, 1]))

        self.assertEqual(set(reset_obs.keys()), set(step_obs.keys()))

    def test_obs_partial_routes_immutable_across_later_steps(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)
        env.reset(seed=0)

        obs_t1, *_ = env.step(np.array([0, 1]))
        captured = obs_t1["partial_routes"]
        env.step(np.array([0, 2]))
        env.step(np.array([1, 3]))

        self.assertEqual(captured, ((0, 1), (0, 0)))

    def test_done_only_true_after_returning_to_depot(self):
        env = HeterogeneousCVRP(n_nodes=3, n_vehicles=1)
        env.reset(seed=0)
        env.demand = np.array(
            [[0.0], [0.1], [0.1], [0.1]], dtype=np.float32
        )

        _, _, done_after_pick1, *_ = env.step(np.array([0, 1]))
        _, _, done_after_pick2, *_ = env.step(np.array([0, 2]))
        _, _, done_after_last_pick, *_ = env.step(np.array([0, 3]))
        _, _, done_after_return, *_ = env.step(np.array([0, 0]))

        self.assertFalse(done_after_pick1)
        self.assertFalse(done_after_pick2)
        self.assertFalse(
            done_after_last_pick,
            "all customers visited but vehicle not at depot yet",
        )
        self.assertTrue(done_after_return)

    def test_observation_space_contains_obs(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)
        reset_obs, _ = env.reset(seed=0)
        self.assertTrue(env.observation_space.contains(reset_obs))

        step_obs, *_ = env.step(np.array([0, 1]))
        self.assertTrue(env.observation_space.contains(step_obs))

    def test_reset_does_not_pollute_global_rng(self):
        np.random.seed(999)
        baseline = np.random.uniform(0, 1, 5)

        np.random.seed(999)
        env = HeterogeneousCVRP(n_nodes=10, n_vehicles=3)
        env.reset(seed=12345)
        after = np.random.uniform(0, 1, 5)

        np.testing.assert_array_equal(baseline, after)


class HCVRPCostTests(unittest.TestCase):
    def test_default_acc_cost_equals_total_distance(self):
        env = HeterogeneousCVRP(n_nodes=3, n_vehicles=1)
        env.reset(seed=0)
        env.node_loc = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        env.demand = np.array(
            [[0.0], [0.1], [0.1], [0.1]], dtype=np.float32
        )

        env.step(np.array([0, 1]))
        env.step(np.array([0, 2]))
        env.step(np.array([0, 3]))
        env.step(np.array([0, 0]))

        # default fc=0, vc=1, speed=1 → acc_cost == acc_travel_time
        np.testing.assert_allclose(
            env.acc_cost.flatten(), env.acc_travel_time.flatten(), atol=1e-5
        )

    def test_fixed_cost_charged_once_on_first_deployment(self):
        env = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=1, fixed_costs=[10.0]
        )
        env.reset(seed=0)
        env.node_loc = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        env.demand = np.array(
            [[0.0], [0.1], [0.1], [0.1]], dtype=np.float32
        )

        self.assertFalse(env.deployed[0])
        env.step(np.array([0, 1]))
        self.assertTrue(env.deployed[0])

        # cost after first move = 10 (fc) + 1.0 (distance × vc=1)
        self.assertAlmostEqual(float(env.acc_cost[0, 0]), 11.0, places=5)

        # subsequent move only adds variable cost (1.0 distance)
        env.step(np.array([0, 2]))
        self.assertAlmostEqual(float(env.acc_cost[0, 0]), 12.0, places=5)

    def test_fixed_cost_not_charged_for_waiting_vehicle(self):
        env = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=2, fixed_costs=[10.0, 20.0]
        )
        env.reset(seed=0)
        env.node_loc = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        env.demand = np.array(
            [[0.0], [0.1], [0.1], [0.1]], dtype=np.float32
        )

        env.step(np.array([0, 1]))

        self.assertTrue(env.deployed[0])
        self.assertFalse(env.deployed[1])
        self.assertAlmostEqual(float(env.acc_cost[0, 0]), 11.0, places=5)
        self.assertEqual(float(env.acc_cost[1, 0]), 0.0)

    def test_variable_cost_scales_distance(self):
        env = HeterogeneousCVRP(
            n_nodes=2, n_vehicles=2,
            capacities=[10.0, 10.0],
            variable_costs=[1.0, 3.0],
            demand_dist="discrete",
        )
        env.reset(seed=0)
        env.node_loc = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )

        env.step(np.array([0, 1]))
        self.assertAlmostEqual(float(env.acc_cost[0, 0]), 1.0, places=5)

        env.reset(seed=0)
        env.node_loc = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        env.step(np.array([1, 1]))
        self.assertAlmostEqual(float(env.acc_cost[1, 0]), 3.0, places=5)

    def test_min_sum_cost_step_rewards_sum_to_negative_total(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2,
            fixed_costs=[5.0, 5.0],
            variable_costs=[2.0, 1.0],
            objective="min_sum_cost",
        )
        env.reset(seed=7)

        total = 0.0
        for action in [(0, 1), (0, 2), (0, 3), (0, 0), (1, 4), (1, 5), (1, 0)]:
            _, r, *_ = env.step(np.array(action))
            total += r

        expected = -float(env.acc_cost.sum())
        self.assertAlmostEqual(total, expected, places=5)

    def test_min_max_cost_zero_until_terminal(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2,
            fixed_costs=[5.0, 5.0],
            objective="min_max_cost",
        )
        env.reset(seed=7)

        rewards = []
        done = False
        for action in [(0, 1), (0, 2), (0, 3), (0, 0), (1, 4), (1, 5), (1, 0)]:
            _, r, done, _, _ = env.step(np.array(action))
            rewards.append(r)

        self.assertTrue(done)
        for r in rewards[:-1]:
            self.assertEqual(r, 0.0)
        self.assertLess(rewards[-1], 0.0)
        self.assertAlmostEqual(
            rewards[-1], -float(np.max(env.acc_cost)), places=5
        )

    def test_reward_method_for_cost_objectives(self):
        env_max = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=2, objective="min_max_cost"
        )
        env_max.reset(seed=0)
        env_max.acc_cost = np.array([[3.0], [7.0]], dtype=np.float32)
        self.assertAlmostEqual(env_max.reward(), -7.0, places=5)

        env_sum = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=2, objective="min_sum_cost"
        )
        env_sum.reset(seed=0)
        env_sum.acc_cost = np.array([[3.0], [7.0]], dtype=np.float32)
        self.assertAlmostEqual(env_sum.reward(), -10.0, places=5)

    def test_obs_includes_vehicle_specs(self):
        env = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=2,
            fixed_costs=[5.0, 10.0],
            variable_costs=[1.5, 2.0],
        )
        obs, _ = env.reset(seed=0)

        np.testing.assert_array_equal(
            obs["fixed_costs"].flatten(), [5.0, 10.0]
        )
        np.testing.assert_array_equal(
            obs["variable_costs"].flatten(), [1.5, 2.0]
        )
        np.testing.assert_array_equal(obs["acc_cost"].flatten(), [0.0, 0.0])
        np.testing.assert_array_equal(obs["deployed"], [0, 0])
        self.assertTrue(env.observation_space.contains(obs))


class HCVRPMultiDepotTests(unittest.TestCase):
    def test_all_depots_marked_visited_at_reset(self):
        env = HeterogeneousCVRP(n_nodes=3, n_vehicles=1, n_depots=4)
        env.reset(seed=0)
        np.testing.assert_array_equal(env.visited[:4], [True, True, True, True])

    def test_episode_terminates_with_n_depots_greater_than_n_vehicles(self):
        env = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=1, n_depots=3, capacities=[1.0]
        )
        env.reset(seed=0)
        env.partial_routes = [[0]]
        env.visited[: env.n_depots] = True
        env.visited[env.n_depots :] = False
        env.demand = np.array(
            [[0.0], [0.0], [0.0], [0.2], [0.2], [0.2]], dtype=np.float32
        )

        env.step(np.array([0, 3]))
        env.step(np.array([0, 4]))
        env.step(np.array([0, 5]))
        _, _, done, _, _ = env.step(np.array([0, 0]))
        self.assertTrue(done)

    def test_capacity_refills_at_non_starting_depot(self):
        env = HeterogeneousCVRP(
            n_nodes=3, n_vehicles=1, n_depots=2, capacities=[1.0]
        )
        env.reset(seed=0)
        env.partial_routes = [[0]]
        env.visited[: env.n_depots] = True
        env.demand = np.array(
            [[0.0], [0.0], [0.5], [0.0], [0.0]], dtype=np.float32
        )

        env.step(np.array([0, 2]))
        cap_after_pickup = float(env.free_capacity[0, 0])
        self.assertLess(cap_after_pickup, 1.0)

        env.step(np.array([0, 1]))
        self.assertEqual(float(env.free_capacity[0, 0]), 1.0)

    def test_vehicle_can_end_at_different_depot_than_start(self):
        env = HeterogeneousCVRP(n_nodes=2, n_vehicles=1, n_depots=2)
        env.reset(seed=0)
        env.partial_routes = [[0]]
        env.visited[: env.n_depots] = True
        env.visited[env.n_depots :] = False
        env.demand = np.array(
            [[0.0], [0.0], [0.2], [0.2]], dtype=np.float32
        )

        env.step(np.array([0, 2]))
        env.step(np.array([0, 3]))
        _, _, done, _, _ = env.step(np.array([0, 1]))

        self.assertTrue(done)
        self.assertEqual(env.partial_routes[0][-1], 1)

    def test_full_episode_with_two_depots_two_vehicles(self):
        env = HeterogeneousCVRP(
            n_nodes=4, n_vehicles=2, n_depots=2, capacities=[1.0, 1.0]
        )
        env.reset(seed=0)
        env.partial_routes = [[0], [1]]
        env.visited[: env.n_depots] = True
        env.visited[env.n_depots :] = False
        env.demand = np.array(
            [[0.0], [0.0], [0.3], [0.3], [0.3], [0.3]], dtype=np.float32
        )

        env.step(np.array([0, 2]))
        env.step(np.array([0, 3]))
        env.step(np.array([0, 0]))
        env.step(np.array([1, 4]))
        env.step(np.array([1, 5]))
        _, _, done, _, _ = env.step(np.array([1, 1]))

        self.assertTrue(done)
        self.assertEqual(env.partial_routes[0][-1], 0)
        self.assertEqual(env.partial_routes[1][-1], 1)

    def test_vehicle_at_any_depot_can_reach_all_other_depots_after_pickup(self):
        env = HeterogeneousCVRP(n_nodes=2, n_vehicles=1, n_depots=3)
        env.reset(seed=0)
        env.partial_routes = [[0]]
        env.visited[: env.n_depots] = True

        env.step(np.array([0, env.n_depots]))
        mask = env.get_action_mask()
        np.testing.assert_array_equal(mask[0, : env.n_depots], [1, 1, 1])


class HCVRPRenderTests(unittest.TestCase):
    def test_render_returns_none_when_mode_none(self):
        env = HeterogeneousCVRP(n_nodes=5, n_vehicles=2)
        env.reset(seed=0)
        self.assertIsNone(env.render())

    def test_render_rejects_invalid_mode(self):
        with self.assertRaises(AssertionError):
            HeterogeneousCVRP(n_nodes=5, n_vehicles=2, render_mode="bogus")

    def test_render_rgb_array_shape_and_dtype(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2, render_mode="rgb_array"
        )
        env.reset(seed=0)
        img = env.render()
        try:
            self.assertIsInstance(img, np.ndarray)
            self.assertEqual(img.dtype, np.uint8)
            self.assertEqual(img.ndim, 3)
            self.assertEqual(img.shape[2], 3)
            self.assertGreater(img.shape[0], 100)
            self.assertGreater(img.shape[1], 100)
        finally:
            env.close()

    def test_render_changes_after_step(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2, render_mode="rgb_array"
        )
        env.reset(seed=0)
        before = env.render()
        env.step(np.array([0, 1]))
        after = env.render()
        try:
            self.assertFalse(np.array_equal(before, after))
        finally:
            env.close()

    def test_render_before_reset_raises(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2, render_mode="rgb_array"
        )
        with self.assertRaises(RuntimeError):
            env.render()

    def test_close_is_idempotent(self):
        env = HeterogeneousCVRP(
            n_nodes=5, n_vehicles=2, render_mode="rgb_array"
        )
        env.reset(seed=0)
        env.render()
        env.close()
        env.close()


if __name__ == "__main__":
    unittest.main()
