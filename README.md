
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests Workflow](https://github.com/HublyGroup/logistics-or-gym/actions/workflows/python-app.yml/badge.svg)
 [![PyPI version](https://badge.fury.io/py/logistics-or-gym.svg)](https://badge.fury.io/py/logistics-or-gym)

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
Python versions supported are: >=3.8 <3.12

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
The arguments you can pass are: 
````python
n_vehicles=3, 
n_nodes=50
````

There will be more arguments later. All fields are public so in the meantime just rewrite the properties

# TODO
- [X] HCVRP
- [ ] Container filling (3D binpacking)
- [ ] Dynamic HCVRP (For delivery)
- [ ] Supply Chain Management (Not yet decided on which ones)
# Credit
