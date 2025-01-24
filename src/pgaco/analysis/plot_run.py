from os.path import dirname

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import pgaco
from pgaco.utils import parallel_runs, plot, get_graph
from pgaco.models import ACO, ACOSGD, ACOPG, ADACO, ANTQ

def cal_total_distance(routine):
            size = len(routine)
            return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                        for i in range(size)])


def run_aco(distance_matrix, seed):
    aco = ACO(distance_matrix,
              minmax=False,
              slim = False,
                 replay_rule = "none",
              seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None, aco._name_

def run_minmaxaco(distance_matrix, seed):
    aco = ACO(distance_matrix,
              slim = False,
            replay_rule = "none",
              seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None,  "MINMAX " + aco._name_

def run_adaco(distance_matrix, seed):
    aco = ADACO(distance_matrix,
                slim = False,
                 replay_rule = "none",
                seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None, aco._name_

def run_antq(distance_matrix, seed):
    aco = ANTQ(distance_matrix,
               slim = False,
                 replay_rule = "none",
               seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None, aco._name_


def run_acosgd(distance_matrix, seed):
    aco = ACOSGD(distance_matrix,
                 slim = False,
                 replay_rule = "none",
                 seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None, aco._name_

def run_acopg(distance_matrix, seed):
    aco = ACOPG(distance_matrix,
                slim = False,
                 replay_rule = "none",
                epsilon = -1,
                seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None, aco._name_

def run_acoppo(distance_matrix, seed):
    aco = ACOPG(distance_matrix,
                slim = False,
                 replay_rule = "none",
                seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None, aco._name_ + " w/ clip"

if __name__ == "__main__":
    """python plot_run.py runs rho alpha beta pop_size graph max_iter learning_rate"""
    # Reading in params
    global max_iter
    global seed
    seed = 42
    max_iter = 50
    runs = 5

    module_path     = dirname(pgaco.__spec__.origin)
    save_dir        = f"{module_path}/results/pgtests"
    graph = 20
    # graph = "ali535.tsp"
    distance_matrix = get_graph(graph)

    G = nx.from_numpy_array(distance_matrix)
    cycle = nx.approximation.simulated_annealing_tsp(G, "greedy")
    cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    plot(cost*np.ones(max_iter).reshape((1, -1)), color="black", label="simulated annealing")

    print("running ACO")
    aco_runs, _, aco_name = parallel_runs(run_aco, runs, distance_matrix, seed)
    plot(aco_runs, color="blue", label=aco_name)

    print("running MMACO")
    aco_runs, _, aco_name = parallel_runs(run_minmaxaco, runs, distance_matrix, seed)
    plot(aco_runs, color="green", label=aco_name)

    print("running ADACO")
    aco_runs,  _,aco_name = parallel_runs(run_adaco, runs, distance_matrix, seed)
    plot(aco_runs, color="red", label=aco_name)

    print("running ANTQ")
    aco_runs,  _,aco_name = parallel_runs(run_antq, runs, distance_matrix, seed)
    plot(aco_runs, color="teal", label=aco_name)

    print("running ACOSGD")
    aco_runs,  _,aco_name = parallel_runs(run_acosgd, runs, distance_matrix, seed)
    plot(aco_runs, color="orange", label=aco_name)

    print("running ACOPG")
    aco_runs, _, aco_name = parallel_runs(run_acopg, runs, distance_matrix, seed)
    plot(aco_runs, color="olive", label=aco_name)

    print("running ACOPPO")
    aco_runs, _, aco_name = parallel_runs(run_acoppo, runs, distance_matrix, seed)
    plot(aco_runs, color="purple", label=aco_name)

    plt.legend()
    plt.tight_layout()
    # plt.show()
    save_file = f"{save_dir}/test1.png"
    print(f"file at {save_file}")
    plt.savefig(save_file)

