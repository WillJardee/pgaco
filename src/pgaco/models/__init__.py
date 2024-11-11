"""
PGACO: Policy Gradient Ant Colony Optimization.

This subpackage implements novel algorithms combining Policy Gradient methods
with Ant Colony Optimization for solving complex optimization problems.

Classes:
--------
ACO : Traditional ACO algorithm
ADACO : Adaptive Gradient ACO algorithm
PGACO_LOG : Policy Gradient ACO with logarithmic gradient.
PGACO_RATIO : Policy Gradient ACO with policy ratio.

Examples
--------
>>> from model.PGACO_LOG import PGACO_LOG
>>> from utils.tsplib_reader import TSPGraph
>>> problem = TSPGraph("att48.tsp")
>>> pgaco = PGACO_LOG(problem, learning_rate=10_000)
>>> best_score, best_solution = pgaco.run(max_iter=200)

Notes
-----
This package is part of ongoing research in combinatorial optimization
and reinforcement learning. For the latest updates and detailed usage
instructions, please refer to the official documentation.
"""

version = "0.2.1"

from pgaco.utils import post_init_decorator
from .acobase import ACOBase
from .aco import ACO
from .acosgd import ACOSGD
from .adaco import ADACO
from .antq import ANTQ
# from .acopg import PGACO

__all__ = ["ACOBase", "ACO", "ACOSGD", "ADACO", "ANTQ", "PGACO", "post_init_decorator"]

