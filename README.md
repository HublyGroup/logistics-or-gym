
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests Workflow](https://github.com/HublyGroup/logistics-or-gym/actions/workflows/python-app.yml/badge.svg?kill_cache=1)
 [![PyPI version](https://badge.fury.io/py/logistics-or-gym.svg?kill_cache=1)](https://badge.fury.io/py/logistics-or-gym?kill_cache=1)

![Logo](img/4x/logo2.png#gh-dark-mode-only)
![LogoBlack](img/4x/logo-black.png#gh-light-mode-only)

# Introduction
Logistics-OR-gym is a "collection" of Open AI Gym environments ment to simualte logistical problems such as routing, 
container filling and supply chain 
# Install
You can install the envs using pypi
````shell
pip install logistics-or-gym
````
Optional extras:
````shell
pip install "logistics-or-gym[viz]"    # matplotlib rendering
pip install "logistics-or-gym[train]"  # stable-baselines3 + sb3-contrib (MaskablePPO baseline)
pip install "logistics-or-gym[rl4co]"  # rl4co (POMO / Attention Model)
````
Python versions supported are: >=3.14

# Available Environments
## Routing
### Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)
HCVRP simulates routing problems when the number of vehicles is >=1 (This means it also covers the case for CVRP if only
that is needed) and different speeds. This implementation follows the one from:
````cite
@article{Li2021,
   abstract = {Existing deep reinforcement learning (DRL) based methods for solving the capacitated vehicle routing problem (CVRP) intrinsically cope with homogeneous vehicle fleet, in which the fleet is assumed as repetitions of a single vehicle. Hence, their key to construct a solution solely lies in the selection of the next node (customer) to visit excluding the selection of vehicle. However, vehicles in real-world scenarios are likely to be heterogeneous with different characteristics that affect their capacity (or travel speed), rendering existing DRL methods less effective. In this paper, we tackle heterogeneous CVRP (HCVRP), where vehicles are mainly characterized by different capacities. We consider both min-max and min-sum objectives for HCVRP, which aim to minimize the longest or total travel time of the vehicle(s) in the fleet. To solve those problems, we propose a DRL method based on the attention mechanism with a vehicle selection decoder accounting for the heterogeneous fleet constraint and a node selection decoder accounting for the route construction, which learns to construct a solution by automatically selecting both a vehicle and a node for this vehicle at each step. Experimental results based on randomly generated instances show that, with desirable generalization to various problem sizes, our method outperforms the state-of-the-art DRL method and most of the conventional heuristics, and also delivers competitive performance against the state-of-the-art heuristic method, i.e., SISR. Additionally, the results of extended experiments demonstrate that our method is also able to solve CVRPLib instances with satisfactory performance.},
   author = {Jingwen Li and Yining Ma and Ruize Gao and Zhiguang Cao and Andrew Lim and Wen Song and Jie Zhang},
   doi = {10.1109/TCYB.2021.3111082},
   journal = {IEEE Transactions on Cybernetics},
   keywords = {Computer architecture,Decoding,Deep reinforcement learning (DRL),Optimization,Reinforcement learning,Routing,Search problems,Vehicle routing,heterogeneous CVRP (HCVRP),min-max objective,min-sum objective.},
   month = {10},
   publisher = {Institute of Electrical and Electronics Engineers Inc.},
   title = {Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem},
   url = {http://arxiv.org/abs/2110.02629 http://dx.doi.org/10.1109/TCYB.2021.3111082},
   year = {2021},
}
````
To use the environment simply use the gym library to create it:

````python
import gymnasium as gym
gym.make("hcvrp-v0")
````

Or instantiate the class directly for full control:

````python
from logistics_or_gym.envs import HeterogeneousCVRP

env = HeterogeneousCVRP(
    n_vehicles=3,
    n_nodes=50,
    n_depots=1,                 # multi-depot supported (n_depots > 1)
    capacities=[20, 25, 30],    # per-vehicle capacity
    speeds=[1/4, 1/5, 1/6],     # per-vehicle speed
    fixed_costs=[10, 12, 15],   # per-vehicle deployment cost (paper 1)
    variable_costs=[1, 1.2, 1.5], # per-vehicle cost per unit distance (paper 1)
    objective="min_max",        # min_max | min_sum | min_max_cost | min_sum_cost
    demand_dist="discrete",     # uniform U(0,1) | discrete {1..9}
    render_mode="human",        # human | rgb_array | None
)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([0, 5])  # (vehicle, node)
env.render()
````

The action is a `MultiDiscrete([n_vehicles, n_depots + n_nodes])` — pick a vehicle
and the node it should visit next. The observation is a `Dict` including an
`action_mask` of valid `(vehicle, node)` pairs.

# Training

The `training/` directory contains a small RL stack (not shipped in the wheel):

- `train_ppo.py` — MaskablePPO baseline (`stable-baselines3` + `sb3-contrib`).
- `train_rl4co.py` — POMO / Attention Model on rl4co's `MTVRPEnv`, with LR
  scheduler, configurable embedding dim, checkpointing, and resume.
- `eval_rl4co.py` — inference-config sweep (greedy / multi-start / sampling) on a
  trained checkpoint.
- `live_viewer.py` — real-time tour rendering during training.

## Benchmark

POMO trained on CVRP-50 (200 epochs, `embed_dim=256`, MultiStepLR), evaluated on
1280 fixed instances (`seed=42`) via `training/eval_rl4co.py`:

| Inference config         | Tour length | Time (1280 inst.) |
| ------------------------ | ----------: | ----------------: |
| greedy (single start)    |      -10.79 |              1.1s |
| POMO (50 starts × 8 aug) |      -10.43 |              3.5s |
| sampling (200 starts)    |  **-10.39** |             10.2s |

The sampling-200 result sits in the LKH-3 optimal band for this distribution, at
milliseconds per instance.

# TODO

- [X] HCVRP (multi-vehicle, multi-depot, per-vehicle speeds + costs)
- [X] matplotlib rendering
- [X] RL training stack (MaskablePPO baseline + rl4co POMO/AM)
- [ ] HFVRP variants — open routes, backhauls, time windows, distance limits (rl4co `MTVRPEnv`)
- [ ] Multi-dimensional capacity (weight + volume + dimensions)
- [ ] Multi-depot CVRP in the rl4co training stack
- [ ] Container filling (3D binpacking)
- [ ] Dynamic HCVRP (For delivery)
- [ ] Supply Chain Management (Not yet decided on which ones)

# Credit
