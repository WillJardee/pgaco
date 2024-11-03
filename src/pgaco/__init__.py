"""
PGACO: Policy Gradient Ant Colony Optimization.

This package implements novel algorithms combining Policy Gradient methods
with Ant Colony Optimization for solving complex optimization problems.

Modules:
--------
model : Core implementation of PGACO algorithms.
utils : Utility functions and helper classes.
analysis : Tools for testing PGACO results.
tuning : Tools for tuning ACO parameters.

Classes:
--------
model.ACO : Traditional ACO algorithm
model.ADACO : Adaptive Gradient ACO algorithm
model.PGACO_LOG : Policy Gradient ACO with logarithmic gradient.
model.PGACO_RATIO : Policy Gradient ACO with policy ratio.

Examples:
---------
>>> from pgaco.model.PGACO_LOG import PGACO_LOG
>>> from utils.tsplib_reader import TSPGraph
>>> problem = TSPGraph("att48.tsp")
>>> pgaco = PGACO_LOG(problem, learning_rate=10_000)
>>> best_score, best_solution = pgaco.run(max_iter=200)

Notes:
------
This package is part of ongoing research in combinatorial optimization
and reinforcement learning. For the latest updates and detailed usage
instructions, please refer to the official documentation.
"""

__author__ = """Will Jardee"""
__email__ = 'willjardee@gmail.com'
__version__ = '0.0.1'


