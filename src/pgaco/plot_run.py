import sys
import threading

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from model.PGACO_LOG import PolicyGradient3ACA
from model.PGACO_RATIO import PolicyGradient4ACA
from model.PGACO_RATIO_CLIP import PolicyGradient5ACA
from model.ACO import ACO_TSP
# from model.PGACA import PolicyGradientACA
def cal_total_distance(routine):
            size = len(routine)
            return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                        for i in range(size)])

def get_graph(name):
    try:
        size = int(name)
        dist_mat = np.random.randint(1, 10, size**2).reshape((size, size))
        return  dist_mat.astype(np.float64)
    except (ValueError, TypeError):
        from utils.tsplib_reader import TSPGraph
        return TSPGraph(f"{name}").graph


if __name__ == "__main__":
    """python plot_run.py runs rho alpha beta pop_size graph max_iter learning_rate"""
    # Reading in params
    iters = 1000

    save_dir = "results/pgtests"

    graph = "att48.tsp"
    distance_matrix = get_graph(graph)

    aco = ACO_TSP(distance_matrix,
                  evap_rate =   0.5,
                  alpha     =   0.7220583863165029,
                  beta      =   4.705037074424084,
                  max_iter  =   iters)

    pgaco1 = PolicyGradient3ACA(distance_matrix,
                                evap_rate           =   0.45363263708110035,
                                learning_rate       =   9893.936558809508,
                                annealing_factor    =   0.044527655588653174,
                                alpha               =   4.975906418434734,
                                beta                =   4.693377743319343,
                                max_iter            =   iters)

    pgaco2 = PolicyGradient4ACA(distance_matrix,
                                evap_rate           =   0.5,
                                learning_rate       =   10_000,
                                annealing_factor    =   0.001,
                                alpha               =   5,
                                beta                =   5,
                                max_iter            =   iters)

    pgaco3 = PolicyGradient5ACA(distance_matrix,
                                evap_rate           =   0.5,
                                learning_rate       =   8640.55059748571,
                                annealing_factor    =   0.01,
                                alpha               =   5,
                                beta                =   5,
                                epsilon             =   0.5,
                                max_iter            =   iters)

    print("running aco")
    aco_score, _ = aco.run()
    print("running pgaco-log")
    pgaco1_score, _ = pgaco1.run()
    print("running pgaco-ratio")
    pgaco2_score, _ = pgaco2.run()
    print("running pgaco-ratio-clip")
    pgaco3_score, _ = pgaco3.run()

    plt.plot(aco.generation_best_Y, label=aco._name_)
    plt.plot(pgaco1.generation_best_Y, label=pgaco1._name_)
    plt.plot(pgaco2.generation_best_Y, label=pgaco2._name_)
    plt.plot(pgaco3.generation_best_Y, label=pgaco3._name_)

    plt.legend()
    # plt.show()
    save_file = f"{save_dir}/{graph}_bayesiantune.png"
    print(f"file at {save_file}")
    plt.savefig(save_file)

