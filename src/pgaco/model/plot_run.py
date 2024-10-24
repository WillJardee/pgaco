import sys
import threading

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from GPGACA import PolicyGradient3ACA, PolicyGradient4ACA, PolicyGradient5ACA
from ACA import ACA_TSP
from PGACA import PolicyGradientACA
def cal_total_distance(routine):
            size = len(routine)
            return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                        for i in range(size)])

def get_graph(name):
    try:
        size = int(name)
        return np.random.randint(1, 10, size**2).reshape((size, size))
    except (ValueError, TypeError):
        from utils.tsplib_reader import TSPGraph
        return TSPGraph(f"{name}").graph

def aco_run(model, params, distance_matrix, cost_func=cal_total_distance,):
    aca = model(func=cal_total_distance,
                          distance_matrix=distance_matrix,
                          params=params)
    if verbose: print(f"running {aca._name_}")
    _, skaco_cost = aca.run()

    plt.plot(aca.generation_best_Y, label=aca._name_)
    if verbose: print(skaco_cost)
    return skaco_cost, aca.generation_best_Y, aca
 

if __name__ == "__main__":
    """python plot_run.py runs rho alpha beta pop_size graph max_iter learning_rate"""
    # Reading in params
    save_dir = "trials"
    global verbose
    verbose = bool(eval(sys.argv[9]))
    runs = int(sys.argv[1])
    params = {
            "rho": float(sys.argv[2]),
            "alpha": float(sys.argv[3]),
            "beta": float(sys.argv[4]),
            "pop_size": int(sys.argv[5]),
            "bias": "inv_weight",
            "learning_rate": float(sys.argv[8]),
            "max_iter": int(sys.argv[7])
        }
    if verbose: print(f"running with options: {sys.argv}")
    
    graph = sys.argv[6]
    distance_matrix = get_graph(graph)

    for run in range(runs):
        aco_run(ACA_TSP, params, distance_matrix)
        aco_run(PolicyGradient3ACA, params, distance_matrix)
        aco_run(PolicyGradient4ACA, params, distance_matrix)
        aco_run(PolicyGradient5ACA, params, distance_matrix)
         

        G = nx.from_numpy_array(distance_matrix, create_using=nx.DiGraph)
        approx = nx.approximation.simulated_annealing_tsp(G, "greedy", source=0)
        plt.plot(np.arange(0, params["max_iter"]), np.ones([params["max_iter"]])*cal_total_distance(approx))

        plt.legend()
        # plt.show()
        save_file = f"{save_dir}/graphsize{graph}_rho{params['rho']}_alpha{params['alpha']}_beta{params['beta']}_pop{params['pop_size']}_learnrate{params['learning_rate']}_{run}.png"
        print(f"file at {save_file}")
        plt.savefig(save_file)
        
        
        
        if verbose: print(cal_total_distance(approx))
        
