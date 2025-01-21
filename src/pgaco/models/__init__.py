"""
PGACO: Policy Gradient Ant Colony Optimization.

This subpackage implements novel algorithms combining Policy Gradient methods
with Ant Colony Optimization for solving complex optimization problems.

Classes:
--------
aco : Traditional ACO algorithm
acosgd : ACO equipped with a Stochastic Gradient Descent update.
adaco : ACOSGD with an ADADELTA adaptive learning-rate.
acopg : Policy Gradient ACO with policy ratio.
antq : ACO with Q-learning update

Examples
--------
>>> from model import acosgd
>>> from utils.tsplib_reader import TSPGraph
>>> problem = TSPGraph("att48.tsp")
>>> acosgd = acosgd(problem,
                   pop_size=2,
                   annealing_factor=0.02)
>>> best_score, best_solution = acosgd.run(max_iter=200)

Notes
-----
This package is part of ongoing research in combinatorial optimization
and reinforcement learning. For the latest updates and detailed usage
instructions, please refer to the official documentation.
"""

version = "0.2.1"

from pgaco.utils import post_init_decorator
from .acobase import ACOBase, path_len
from .aco import ACO
from .acosgd import ACOSGD
from .adaco import ADACO
from .antq import ANTQ
from .acopg import ACOPG

__all__ = ["ACOBase",
           "ACO",
           "ACOSGD",
           "ADACO",
           "ANTQ",
           "ACOPG",
           "post_init_decorator",
           "path_len"]

