from gymnasium.envs.registration import register

register(
    id="hcvrp-v0",
    entry_point="routing.envs:HCVRP",
)
