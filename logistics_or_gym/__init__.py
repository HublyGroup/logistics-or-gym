from gymnasium.envs.registration import register

register(
    id="hcvrp-v0",
    entry_point="logistics_or_gym.envs:HeterogeneousCVRP",
)
