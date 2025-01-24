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
    aco = MODEL(distance_matrix,
                advantage_func = current_adv_func,
                minmax=True,
                slim = False,
                evap_rate = 100,
                size_pop = 10,
                replay_size = 30,
                beta = 1,
                exact_grad = True,
                seed = seed)
    aco.run(MAX_ITER)
    return aco.generation_best_Y, None, aco._name_



ADV_FUNCS_COLORS = {
    # "local": "blue",               # Standard blue
    # "path": "green",               # Standard green
    # "quality": "red",              # Standard red
    "reward": "orange",            # Orange for distinction
    "reward-to-go": "purple",      # Purple for uniqueness
    "reward-to-go-baseline": "brown",  # Brown for muted contrast
    "reward-baseline": "pink",     # Light pink
    # "reward-decay": "cyan",        # Bright cyan
    # "reward-entropy": "magenta",   # Bold magenta
}


ADV_FUNCS = list(ADV_FUNCS_COLORS.keys())

if __name__ == "__main__":
    """python plot_run.py runs rho alpha beta pop_size graph max_iter learning_rate"""
    # Reading in params
    MODEL = ACOSGD


    global MAX_ITER
    global SEED
    global current_adv_func
    SEED = 42
    MAX_ITER = 100
    runs = 5

    module_path     = dirname(pgaco.__spec__.origin)
    save_dir        = f"{module_path}/results/pgtests"
    # graph = 10
    # graph = "ali535.tsp"
    graph = "att48.tsp"
    distance_matrix = get_graph(graph)

    G = nx.from_numpy_array(distance_matrix)
    cycle = nx.approximation.simulated_annealing_tsp(G, "greedy")
    cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    plot(cost*np.ones(MAX_ITER).reshape((1, -1)), color="black", label="simulated annealing")

    for func in ADV_FUNCS:
        current_adv_func = func
        print(f"running {func}")
        aco_runs, _, aco_name = parallel_runs(run_aco, runs, distance_matrix, SEED)
        plot(aco_runs, color=ADV_FUNCS_COLORS[func], label=func)

    plt.legend()
    plt.tight_layout()
    # plt.show()
    save_file = f"{save_dir}/advfunc_test1.png"
    print(f"file at {save_file}")
    plt.savefig(save_file)

