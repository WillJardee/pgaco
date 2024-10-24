"""Main runner file for testing my ACO_TSP."""
import time
from multiprocessing.dummy import Pool
from cProfile import Profile
from pstats import SortKey, Stats
from sko.ACA import ACA_TSP
import networkx as nx


def test_tsp_eq(sol1, sol2):
    """Simple method to test if two ordered list have the same cyclic order."""
    return set([(sol1[i], sol1[i+1])
                for i in range(len(sol1)-1)]) == set([(sol2[i], sol2[i+1])
                                                      for i in range(len(sol2)-1)])


def _SA_method(G, wt):
    SA_tsp = nx.approximation.simulated_annealing_tsp
    return SA_tsp(G, "greedy", weight=wt, temp=500)


def main(size: int, params: dict = {}):
    """Primary runner for evaulating solutions to TSP."""
    str_builder = f"TSP on complete graph (random weights) of size {size}\n"
    G = Graph(nx.complete_graph(size).to_directed())

    # My ACO
    aco = AntColony(graph=G, params=params)
    aco.run()
    aco_sol = aco.best_path()
    aco_cost = sum([G.graph[i[0]][i[1]]['weight'] for i in
                    [(aco_sol[i], aco_sol[i+1]) for i in range(len(aco_sol)-1)]])
    str_builder += f"My ACO - {aco_sol}: {aco_cost}\n"

    # SA
    SA_sol = nx.approximation.traveling_salesman_problem(G.graph,
                                                         method=_SA_method)
    SA_cost = sum([G.graph[i[0]][i[1]]['weight'] for i in
                   [(SA_sol[i], SA_sol[i+1]) for i in range(len(SA_sol)-1)]])
    str_builder += f"nx SA - {SA_sol}: {SA_cost}\n"

    # Scikit-opt ACO
    distance_matrix = nx.adjacency_matrix(G.graph)

    def cal_total_distance(routine):
        return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                    for i in range(size)])

    aca = ACA_TSP(func=cal_total_distance, n_dim=size,
                  size_pop=50, max_iter=200,
                  distance_matrix=distance_matrix)
    skaco_sol, skaco_cost = aca.run()
    str_builder += f"SK ACO - {skaco_sol}: {skaco_cost}\n"

    return str_builder


def collect_perf(runs: int = 10,
                 size: int = 5):
    """Run my ACO once, printing the cProfile."""
    with Profile() as profile:
        times = {}
        for i in range(runs):
            times[i] = time.time()
            str_builder = f"TSP on complete graph (random weights) of size {size}\n"
            G = Graph(nx.complete_graph(size).to_directed())

            # My ACO
            aco = AntColony(graph=G,
                            params={"alpha": 1,
                                    "beta": 2,
                                    "evap_rate": 0.9,
                                    "phero_min": 0.01,
                                    "num_ants": 50,
                                    "ant_max_steps": 100,
                                    "num_iter": 200,
                                    })

            aco.run()
            aco_sol = aco.best_path()
            aco_cost = sum([G.graph[i[0]][i[1]]['weight'] for i in
                            [(aco_sol[i], aco_sol[i+1]) for i in
                             range(len(aco_sol)-1)]])
            str_builder += f"My ACO - {aco_sol}: {aco_cost}"
            print(str_builder)
            times[i] = time.time() - times[i]

    for run, t in times.items():
        print(f"Run {run} took {t:.4f} s")
    Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()


def parallel_runs(runs: int = 10,
                  params: dict = {},
                  size: int = 5):
    """Run main runs times with the default number of theads (from multiprocessing)."""
    with Pool() as pool:
        results = pool.imap_unordered(main, [size]*runs)
        for test in results:
            print(test)


if __name__ == "__main__":
    from model.aco_tsp import PolicyGradientACO as AntColony
    from utils.graph import Graph
    print(parallel_runs(size=100, params={"alpha": 1,
                                          "beta": 2,
                                          "evap_rate": 0.9,
                                          "phero_min": 0.01,
                                          "num_ants": 50,
                                          "ant_max_steps": 200,
                                          "num_iter": 100,
                                          "advantage_type": "forward"}))
