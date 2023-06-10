from logistics_or_gym.envs import HCVRP

# Parallel environments
n_vehicles = 3
n_nodes = 50

env = HCVRP(n_vehicles=n_vehicles, n_nodes=n_nodes)

