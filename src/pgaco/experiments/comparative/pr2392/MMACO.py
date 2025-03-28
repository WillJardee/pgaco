from os.path import dirname

import numpy as np

import pgaco
from pgaco.utils import parallel_runs, plot, get_graph
from pgaco.models import ACO, ACOSGD, ACOPG, ADACO, ANTQ

def cal_total_distance(routine):
            size = len(routine)
            return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                        for i in range(size)])


def run_model(distance_matrix, seed):
    aco = MODEL(distance_matrix,
                slim = False,
                replay_rule = "global_best",
                seed = seed)
    aco.run(max_iter)
    return aco.generation_best_Y, None, aco._name_

if __name__ == "__main__":
    """python plot_run.py runs rho alpha beta pop_size graph max_iter learning_rate"""
    # Reading in params
    global max_iter
    global seed
    seed = 47
    max_iter = 9000
    runs = 10

    MODEL = ACO
    name = "MMACO"

    graph = "pr2392.tsp"
    graph_name = "pr2392"
    distance_matrix = get_graph(graph)

    module_path     = dirname(pgaco.__spec__.origin)
    save_dir        = f"{module_path}/experiments/comparative/results/"
    run_name = save_dir + "_".join([graph_name, name, str(max_iter)]) + ".npy"
    # graph = 20
    # graph = "ali535.tsp"
    # graph = "pcb442.tsp"

    print(f"Running {graph_name} {name}")
    aco_runs, _, aco_name = parallel_runs(run_model, runs, distance_matrix, seed)
    np.save(run_name, aco_runs)

    print(f"Saved to {run_name}")

