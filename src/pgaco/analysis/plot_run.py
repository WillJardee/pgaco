import multiprocessing
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pgaco.model.ACO import ACO_TSP
from pgaco.model.PGACO_LOG import PGACO_LOG
from pgaco.model.PGACO_RATIO import PGACO_RATIO
from pgaco.model.ADACO import ADACO

def cal_total_distance(routine):
            size = len(routine)
            return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                        for i in range(size)])

def get_graph(name):
    try:
        size = int(name)
        dist_mat = np.random.randint(1, 100, size**2).reshape((size, size))
        return  dist_mat.astype(np.float64)
    except (ValueError, TypeError):
        from pgaco.utils.tsplib_reader import TSPGraph
        return TSPGraph(f"{name}").graph

def plot(runs:np.ndarray, color:str, label:str, alpha: float = 0.3) -> None:
    runs = np.array(runs)
    aco_mean = np.mean(runs, axis = 0)
    aco_std = np.std(runs, axis=0)
    time_steps = np.arange(len(aco_mean))
    plt.plot(aco_mean, color=color, label=label)
    plt.fill_between(time_steps,
                     aco_mean - aco_std,
                     aco_mean + aco_std,
                     alpha=alpha, color=color)

def run_aco(distance_matrix, seed):
    aco = ACO_TSP(distance_matrix,
                  evap_rate=0.1,
                  alpha=1,
                  beta=2,
                  max_iter=max_iter,
                  max_tau=-1,
                  min_tau=-1,
                  minmax=False,
                  seed = seed)
    aco.run()
    return aco.generation_best_Y, aco._name_

def run_minmaxaco(distance_matrix, seed):
    aco = ACO_TSP(distance_matrix,
                  evap_rate     =   0.3281659539619214,
                  alpha         =   4,
                  beta          =   8,
                  size_pop      =   100,
                  # size_pop      =   10,
                  replay_size   =   -1,
                  # replay_size   =   10,
                  max_iter      =   max_iter,
                  seed          =   seed)
    aco.run()
    return aco.generation_best_Y, "MINMAX " + aco._name_ + " bayesian"

def run_minmaxaco_hand(distance_matrix, seed):
    aco = ACO_TSP(distance_matrix,
                  evap_rate =   0.1,
                  alpha     =   1,
                  beta      =   2,
                  max_iter  =   max_iter,
                  seed      =   seed)
    aco.run()
    return aco.generation_best_Y, "MINMAX " + aco._name_ + " hand"


def run_adaco(distance_matrix, seed):
    aco = ADACO(distance_matrix,
                evap_rate =   0.1,
                alpha     =   1,
                beta      =   2,
                max_iter  =   max_iter,
                seed      =   seed)
    aco.run()
    return aco.generation_best_Y, aco._name_


def run_pgaco1(distance_matrix, seed):
    aco = PGACO_LOG(distance_matrix,
                    evap_rate           =   0.1,
                    learning_rate       =   1_00,
                    annealing_factor    =   0.01,
                    alpha               =   1,
                    beta                =   2,
                    max_iter            =   max_iter,
                    seed                =   seed)
    aco.run()
    return aco.generation_best_Y, aco._name_

def run_pgaco2(distance_matrix, seed):
    aco = PGACO_RATIO(distance_matrix,
                      evap_rate           =   0.1,
                      learning_rate       =   1_000,
                      annealing_factor    =   0.01,
                      alpha               =   1,
                      beta                =   2,
                      max_iter            =   max_iter,
                      epsilon             =   -1,
                      seed                =   seed) # disables the clipping
    aco.run()
    return aco.generation_best_Y, aco._name_

def run_pgaco3(distance_matrix, seed):
    aco = PGACO_RATIO(distance_matrix,
                      evap_rate           =   0.1,
                      learning_rate       =   1_000,
                      annealing_factor    =   0.01,
                      alpha               =   1,
                      beta                =   2,
                      epsilon             =   0.05,
                      max_iter            =   max_iter,
                      seed                =   seed )
    aco.run()
    return aco.generation_best_Y, aco._name_ + " w/ clip"

def parallel_aco(alg, runs, distance_matrix):
    with multiprocessing.Pool() as pool:
        run_func = partial(alg, distance_matrix)
        results = pool.map(run_func, range(runs))

    aco_runs = [result[0] for result in results]
    aco_name = results[0][1]  # Assuming all runs have the same name
    return np.array(aco_runs), aco_name

if __name__ == "__main__":
    """python plot_run.py runs rho alpha beta pop_size graph max_iter learning_rate"""
    # Reading in params
    global max_iter
    max_iter = 50
    runs = 10

    save_dir = "results/pgtests"
    graph = 10
    distance_matrix = get_graph(graph)

    print("running ACO")
    aco_runs, aco_name = parallel_aco(run_aco, runs, distance_matrix)
    plot(aco_runs, color="green", label=aco_name)

    print("running PGACO LOG")
    aco_runs, aco_name = parallel_aco(run_pgaco1, runs, distance_matrix)
    plot(aco_runs, color="blue", label=aco_name)

    print("running PGACO RATIO")
    aco_runs, aco_name = parallel_aco(run_pgaco2, runs, distance_matrix)
    plot(aco_runs, color="orange", label=aco_name)

    print("running ADACO")
    aco_runs, aco_name = parallel_aco(run_adaco, runs, distance_matrix)
    plot(aco_runs, color="red", label=aco_name)

    plt.legend()
    plt.tight_layout()
    # plt.show()
    save_file = f"{save_dir}/{graph}_test.png"
    print(f"file at {save_file}")
    plt.savefig(save_file)

