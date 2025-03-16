from enum import Enum
import line_profiler
import numpy as np
from numpy.typing import NDArray
import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from pgaco.efficient_rewrite import Aco
from pgaco.efficient_rewrite.prebuilt import *


def plot_cycles(G, cycle1, cycle2, pos=None):
    """
    Plot a NetworkX graph highlighting two cycles with different colors.

    Parameters:
    - G (nx.Graph): The graph to plot.
    - cycle1 (list): List of nodes in the first cycle.
    - cycle2 (list): List of nodes in the second cycle.
    - pos (dict, optional): Precomputed positions for nodes.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)  # Use a consistent layout

    plt.figure(figsize=(8, 6))

    # Draw the base graph
    nx.draw(
        G, pos, node_color="lightgray", node_size=300,
        alpha=0, with_labels=True
    )

    # Draw first cycle (e.g., red)
    cycle1_edges = [(cycle1[i], cycle1[i+1]) for i in range(len(cycle1)-1)] + [(cycle1[-1], cycle1[0])]
    nx.draw_networkx_edges(
        G, pos, edgelist=cycle1_edges,
        edge_color="black", width=2, style="dashed", label="SA"
    )

    # Draw second cycle (e.g., blue)
    cycle2_edges = [(cycle2[i], cycle2[i+1]) for i in range(len(cycle2)-1)] + [(cycle2[-1], cycle2[0])]
    nx.draw_networkx_edges(
        G, pos, edgelist=cycle2_edges,
        edge_color="grey", width=2, style="solid", label="ACO"
    )

    plt.legend()
    plt.title("Two Cycles on a Graph")
    plt.show()


if __name__ == "__main__":
    # from pgaco.utils import get_graph, plot, parallel_runs
    # graph = get_graph("berlin52.tsp")
    # G = nx.from_numpy_array(graph)

    # import datetime
    # now = datetime.datetime.now()
    # print(now.strftime("%Y-%m-%d %I:%M %p"))
    ITERS = 100
    POP_SIZE = 100
    # SEED = random.randint(0, int(1e6))
    SEED = 42

    print("ITERS: ", ITERS)
    print("SEED: ", SEED)

    import tsplib95
    # problem = tsplib95.load('../tsplib/berlin52.tsp')
    graph_file = '../tsplib/att48.tsp'
    opt = 10628
    # graph_file = '../tsplib/a280.tsp'
    # opt = 2579
    # graph_file = '../tsplib/ch150.tsp'
    # opt = 6528
    problem = tsplib95.load(graph_file)
    G = problem.get_graph()
    graph = nx.to_numpy_array(G)

    print("Avg Dist:", round(graph.mean(),2), "+/-", round(graph.std(),2))

    try:
        plt.axhline(opt, color="black")
        print("opt: ", opt)
    except:
        pass

    # cost_lis = [nx.approximation.greedy_tsp(G, source=i+1) for i in range(graph.shape[0])]
    # cost = np.array([sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle)) for cycle in cost_lis])
    # print("Greedy:", round(cost.mean(),2), "+/-", round(cost.std(),2))
    # plt.axhline(cost.mean(), color="grey", linestyle="--", label="Greedy")
    # cycle = nx.approximation.simulated_annealing_tsp(G, "greedy", max_iterations=ITERS)
    # cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    # cost_lis = [nx.approximation.simulated_annealing_tsp(G, lis, temp=500, max_iterations=ITERS) for lis in cost_lis]
    # cost = np.array([sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle)) for cycle in cost_lis])
    # print("SA:", round(cost.mean(),2), "+/-", round(cost.std(),2))
    # plt.axhline(cost.mean(), color="grey", linestyle="--", label="SA")
    # cost_lis = [nx.approximation.threshold_accepting_tsp(G, lis, max_iterations=ITERS) for lis in cost_lis]
    # cost = np.array([sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle)) for cycle in cost_lis])
    # print("Thresh:", round(cost.mean(),2), "+/-", round(cost.std(),2))
    # plt.axhline(cost.mean(), color="grey", linestyle="--", label="Thresh")
    # cost_lis = [nx.approximation.traveling_salesman_problem(G)]
    # cost = np.array([sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle)) for cycle in cost_lis])
    # print("Christofides:", round(cost.mean(),2), "+/-", round(cost.std(),2))
    # plt.axhline(cost.mean(), color="blue", linestyle="--", label="Christofides")


    # from sko.GA import GA_TSP
    # def cal_total_distance(routine):
    #     '''The objective function. input routine, return total distance.
    #     cal_total_distance(np.arange(num_points))
    #     '''
    #     num_points, = routine.shape
    #     return sum([graph[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    # ga_tsp = GA_TSP(func=cal_total_distance, n_dim=graph.shape[0], size_pop=POP_SIZE, max_iter=ITERS, prob_mut=0.7)
    # best_points, best_distance = ga_tsp.run()
    # print("GA:", best_distance)
    #
    # # best_points_ = np.concatenate([best_points, [best_points[0]]])
    # # best_points_coordinate = points_coordinate[best_points_, :]
    # plt.plot(ga_tsp.generation_best_Y, label="GA")


    aco_list = []
    # aco_list.append(Aco(graph,
    #                     alpha = 1, # policy weight
    #                     beta = 2, # heur weight
    #                     rho = 1000, # 1 - evap rate
    #                     pop_size = 10, # population size
    #                     greedy_epsilon = 0.1, # Greedy perc
    #                     clip_epsilon = 0.2, # Greedy perc
    #                     maxmin = True,
    #                     maxmin_adaptive = True,
    #                     buffer_size = 35,
    #                     buffer_rule = "elite",
    #                     kernel = "identity",
    #                     adv_func = "reward",
    #                     importance_sampling=False,
    #                     seed = SEED,
    #                     gradient = "adaco",
    #                     ))


    # aco_list.append(Aco(graph,
    #                     alpha = 1, # policy weight
    #                     beta = 0, # heur weight
    #                     rho = 0.1, # 1 - evap rate
    #                     pop_size = 10, # population size
    #                     greedy_epsilon = 0.1, # Greedy perc
    #                     clip_epsilon = 0.2, # Greedy perc
    #                     maxmin = True,
    #                     maxmin_adaptive = True,
    #                     buffer_size = 10,
    #                     buffer_rule = "elite",
    #                     kernel = "identity",
    #                     adv_func = "reward",
    #                     importance_sampling=False,
    #                     seed = SEED,
    #                     gradient = "AS",
    #                     ))

    # aco_list.append(Aco(graph,
    #                     alpha = 1, # policy weight
    #                     beta = 0, # heur weight
    #                     rho = 50, # 1 - evap rate
    #                     pop_size = 10, # population size
    #                     greedy_epsilon = 0.1, # Greedy perc
    #                     clip_epsilon = 0.2, # Greedy perc
    #                     maxmin = False,
    #                     maxmin_adaptive = False,
    #                     buffer_size = 35,
    #                     buffer_rule = "elite",
    #                     kernel = "exp",
    #                     importance_sampling=True,
    #                     seed = SEED,
    #                     gradient = "ppo",
    #                     ))

    # aco_list.append(Aco(graph,
    #                     alpha = 1, # policy weight
    #                     beta = 0, # heur weight
    #                     rho = 50, # 1 - evap rate
    #                     pop_size = 10, # population size
    #                     greedy_epsilon = 0.5, # Greedy perc
    #                     clip_epsilon = 0.2, # Greedy perc
    #                     maxmin = False,
    #                     maxmin_adaptive = True,
    #                     buffer_size = 35,
    #                     buffer_rule = "elite",
    #                     kernel = "exp",
    #                     importance_sampling=True,
    #                     seed = SEED,
    #                     gradient = "pg",
    #                     ))

    beta = 0
    # aco_list.append(Acosga(graph, alpha=0.8, beta=beta, buffer_size=30, rho=0.0001))
    # aco_list.append(Adaco(graph, beta=beta, learning_rate=0.01, pop_size=30))
    # aco_list.append(Pgaco(graph,alpha=2, beta=beta, buffer_size=30, rho=1))
    # aco_list.append(Ppoaco(graph,alpha=1,greedy_epsilon=0,  pop_size=17, beta=beta, buffer_size=20, rho=5))
    # aco_list.append(Ppoaco(graph, alpha=1, greedy_epsilon=0.0, clip_epsilon=0.2, beta=beta, rho=0.05, adv_func="reward_baselined"))
    # aco_list.append(Ppoaco(graph,
    #                        alpha=1,
    #                        beta=beta,
    #                        rho=0.1,
    #                        greedy_epsilon=0.3,
    #                        clip_epsilon=0.1,
    #                        buffer_size=30,
    #                        buffer_draws=40,
    #                        pop_size=60,
    #                        adv_func="reward_baselined_to_go"
    #                        )
    #                 )
    aco_list.append(Ppoaco(graph,
                           alpha=1,
                           beta=beta,
                           rho=0.1,
                           greedy_epsilon=0.2,
                           clip_epsilon=np.inf,
                           buffer_size=10,
                           buffer_draws=30,
                           pop_size=30,
                           # buffer_rule="evict",
                           adv_func="reward_baselined"
                           )
                    )
    # aco_list.append(As(graph, beta=beta,greedy_epsilon=0.2, pop_size=30))


    # for i in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
    # for i in ["AS", "maxmin", "acosga", "adaco", "pg", "ppo", "tr_ppo_rb"]:
    for aco in aco_list:
    # for aco, name in zip(aco_list, ["acosga"]):
        aco_cycle, score = aco.run(max_epochs=ITERS)
        # plt.plot(aco.gen_sol_score, label=f"{aco.name}: Best Solution")
        # plt.plot(aco.gen_pol_score, label=f"{i}: Policy Solution")
        print(f"{aco.name} search best: ", score)
        print(f"{aco.name} policy + heur: ", [aco.greedy_solution(heur=True)[1] for _ in range(aco.n)])
        print(f"{aco.name} policy:        ", [aco.greedy_solution(heur=False)[1] for _ in range(aco.n)])
        print()
        plt.plot(np.mean(aco.score_trace_no_heur, axis=1), label=aco.name + " no heur")
        # plt.plot(np.mean(aco.score_trace_heur, axis=1), label=aco.name + " heur")
        # plt.plot(aco.gen_sol_score_policy, label=aco.name)
        # plt.plot(aco.gen_sol_score, label=aco.name)

    plt.legend()
    plt.tight_layout()
    plt.show()
    # pos = {node: data['coord'] for node, data in G.nodes(data=True)}
    # plot_cycles(G, cycle[:-2], (aco_cycle+1)[:-2], pos=pos)




