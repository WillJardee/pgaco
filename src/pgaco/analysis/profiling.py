#!/usr/bin/env python

import cProfile
import pstats
from pstats import SortKey

from pgaco.models import ACOSGD
from pgaco.models.__legacy import ACOSGD as ACOSGDOld
from pgaco.models.__legacy import ACO as ACOOld
from pgaco.models import ACO
from pgaco.utils import get_graph


def run_model1(distance_matrix, seed):
    aco = ACOSGD(distance_matrix,
                 size_pop      = 5,
                 seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_

def run_model2(distance_matrix, seed):
    aco = ACOSGDOld(distance_matrix,
                 size_pop      = 5,
                 seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_


# def run_model1(distance_matrix, seed):
#     aco = ACO(distance_matrix,
#                  size_pop      = 5,
#                  seed          = seed)
#     aco.run(max_iter=max_iter)
#     return aco.generation_best_Y, aco.generation_policy_score, aco._name_
#
# def run_model2(distance_matrix, seed):
#     aco = ACOOld(distance_matrix,
#                  size_pop      = 5,
#                  seed          = seed)
#     aco.run(max_iter=max_iter)
#     return aco.generation_best_Y, aco.generation_policy_score, aco._name_



if __name__ == "__main__":

    with open('profile_results.txt', 'w') as var:
        graph = "att48.tsp"
        max_iter = 15
        distance_matrix = get_graph(graph)

        print(f"Running on: {graph}")
        var.write(f"Running on: {graph}\n")

        print("Running sparse")
        cProfile.run('run_model1(distance_matrix, 42)', 'output.prof')
        p = pstats.Stats('output.prof', stream=var)
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

        print("Running reconstruction")
        cProfile.run('run_model2(distance_matrix, 42)', 'output.prof')
        p = pstats.Stats('output.prof', stream=var)
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

        graph = "ali535.tsp"
        max_iter = 15
        distance_matrix = get_graph(graph)

        var.write(f"Running on: {graph}\n")

        print("Running sparse")
        cProfile.run('run_model1(distance_matrix, 42)', 'output.prof')
        p = pstats.Stats('output.prof', stream=var)
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

        print("Running reconstruction")
        cProfile.run('run_model2(distance_matrix, 42)', 'output.prof')
        p = pstats.Stats('output.prof', stream=var)
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

