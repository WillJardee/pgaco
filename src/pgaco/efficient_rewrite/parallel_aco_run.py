import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import tsplib95

from pgaco.efficient_rewrite import Aco
from pgaco.efficient_rewrite.prebuilt import *

def run_single_instance_reward(graph):
    """Runs a single instance of grad_bandit in a separate process."""
    aco = Ppoaco(graph,
                 alpha=1,
                 beta=0,
                 rho=0.1,
                 greedy_epsilon=0.5,
                 clip_epsilon=0.2,
                 buffer_size=100,
                 buffer_draws=20,
                 pop_size=10,
                 buffer_rule="evict",
                 adv_func="adv_path"
                 )
    _, score = aco.run(start=0, max_epochs=100)

    # return np.mean([aco.greedy_solution(heur=False)[1] for _ in range(aco.n)])
    return aco

def run_single_instance_to_go(graph):
    """Runs a single instance of grad_bandit in a separate process."""
    aco = Ppoaco(graph,
                           alpha=1,
                           beta=0,
                           rho=0.1,
                           greedy_epsilon=0.3,
                           clip_epsilon=0.2,
                           buffer_size=100,
                           buffer_draws=20,
                           pop_size=20,
                           # buffer_rule="evict",
                           adv_func="reward_baselined"
                           )

    _, _ = aco.run(start=0, max_epochs=50)
    # return np.mean([aco.greedy_solution(heur=False)[1] for _ in range(aco.n)])
    return aco.score_trace_no_heur



def main():
    graph_file = "../tsplib/att48.tsp"
    opt = 10628
    problem = tsplib95.load(graph_file)
    G = problem.get_graph()
    graph = nx.to_numpy_array(G)

    num_trials = 1

    with ProcessPoolExecutor() as executor:
        results_baselined = list(tqdm(executor.map(run_single_instance_reward, [graph] * num_trials), total=num_trials))
    # with ProcessPoolExecutor() as executor:
    #     results_to_go = list(tqdm(executor.map(run_single_instance_to_go, [graph] * num_trials), total=num_trials))

    # print(np.mean(results), np.std(results))
    for aco in results_baselined:
        # plt.plot(path, label="reward_baselined", color="red")
        plt.plot(np.mean(aco.score_trace_no_heur, axis=1), label=aco.name + " no heur")
        plt.plot(np.mean(aco.score_trace_heur, axis=1), label=aco.name + " heur")
    # for path in results_to_go:
    #     plt.plot(path, label="reward_baselined_to_go", color="blue")
    # print(results_baselined)
    # plt.plot(results_baselined, label="reward_baselined", color="blue")
    # plt.plot(results_baselined[-1], label="reward_baselined", color="blue")
    # plt.hist(results, bins=30)
    # plt.vlines(opt, 0, 40)
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
