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

def aco_run(model, distance_matrix, **kwargs):
    aca = model(distance_matrix=distance_matrix, **kwargs)
    if verbose: print(f"running {aca._name_}")
    _, skaco_cost = aca.run()

    plt.plot(aca.generation_best_Y, label=aca._name_)
    if verbose: print(skaco_cost)
    return skaco_cost, aca.generation_best_Y, aca


if __name__ == "__main__":
    """python plot_run.py runs rho alpha beta pop_size graph max_iter learning_rate"""
    # Reading in params
    save_dir = "results/pgtests"
    global verbose
    verbose = bool(eval(sys.argv[9]))
    runs = int(sys.argv[1])
    params = {
              "evap_rate": float(sys.argv[2]),
              "alpha": float(sys.argv[3]),
              "beta": float(sys.argv[4]),
              "size_pop": int(sys.argv[5]),
              "learning_rate": float(sys.argv[8]),
              "max_iter": int(sys.argv[7])
    }
    if verbose: print(f"running with options: {sys.argv}")

    graph = sys.argv[6]
    distance_matrix = get_graph(graph)

    for run in range(runs):
        aco_run(ACO_TSP, distance_matrix, **params)
        aco_run(PolicyGradient3ACA, distance_matrix, **params)
        aco_run(PolicyGradient4ACA, distance_matrix, **params)
        aco_run(PolicyGradient5ACA, distance_matrix, **params)


        G = nx.from_numpy_array(distance_matrix + 10 * np.eye(distance_matrix.shape[0]), create_using=nx.DiGraph)
        approx = nx.approximation.simulated_annealing_tsp(G, "greedy", source=0)
        plt.plot(np.arange(0, params["max_iter"]), np.ones([params["max_iter"]])*cal_total_distance(approx))

        plt.legend()
        # plt.show()
        save_file = f"{save_dir}/graphsize{graph}_rho{params['evap_rate']}_alpha{params['alpha']}_beta{params['beta']}_pop{params['size_pop']}_learnrate{params['learning_rate']}_{run}.png"
        print(f"file at {save_file}")
        plt.savefig(save_file)



        if verbose: print(cal_total_distance(approx))

