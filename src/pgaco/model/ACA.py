#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/11
# @Author  : github.com/guofei9987

# -*- coding: utf-8 -*-
# @Time    : 2024/4/2
# @Author  : github.com/willjardee


# pylint: disable=line-too-long

import pickle
import ast

import numpy as np
from tqdm import tqdm
import networkx as nx


class ACA_TSP:
    """Simple, default ACA solution to TSP problem."""

    def __init__(self,
                 func  = None,
                 distance_matrix: np.ndarray | None = None,
                 params: dict = {}) -> None:
        """
        Builder for base ACA class.

        Args:
            func = function to optimize wrt.
            distance_matrix = weighted adjacency matrix of the given TSP space.
            params: dict = any non-default params
        """
        self.distance_matrix = distance_matrix # cost matrix
        self.n_dim = distance_matrix.shape[0] if distance_matrix is not None else 0
        self.func = func # optimization function
        self.size_pop = params.get("size_pop", 10)
        self.max_iter = params.get("max_iter", 20)
        self.alpha = params.get("alpha", 1)
        self.beta = params.get("beta", 2)
        self.rho = params.get("rho", 0.1)
        self.min_dist = self.distance_matrix.min() if self.distance_matrix is not None else 0
        self.max_dist = self.distance_matrix.max() if self.distance_matrix is not None else np.inf
        self.min_tau = params.get("min_tau", 0.001 * self.min_dist)
        self.prob_bias = params.get("bias", "")
        self.lamb = params.get("branching_factor", 0.2)
        self.metric_branching_factor = False
        self.save_file = params.get("save_file", None)
        self._checkpoint_file = params.get("checkpoint_file", None)
        self._checkpoint_res = params.get("checkpoint_res", 10)
        if self.distance_matrix is not None:
            self._build_workspace()
        self._name_ = "ACA"

    def _build_workspace(self):
        # building workspace
        self.Tau = np.ones((self.n_dim, self.n_dim))
        self.distance_matrix += 1e-10 * np.eye(self.n_dim)

        # set probability bias
        match self.prob_bias.lower():
            case "": # default is uniform
                self.prob_bias = np.ones((self.distance_matrix.shape))
            case "inv_weight": # inverse distance
                # self.prob_bias = 1 / (self.distance_matrix + 1e-10 * np.eye(self.n_dim, self.n_dim))  # bias value (1/len)
                self.prob_bias = 1 / (self.distance_matrix) # bias value (1/len)
                self.prob_bias[np.where(self.prob_bias == np.inf)] = 0
            case _:
                raise ValueError(f"Invalid bias value of {self.prob_bias}")       # self.prob_bias = 1 / (self.distance_matrix + 1e-10 * np.eye(self.n_dim, self.n_dim))  # bias value (1/len)

        self.Table = np.zeros((self.size_pop, self.n_dim)).astype(int)  # set of ant solutions; row - ant in order of visit
        self.generation_best_X, self.generation_best_Y = [], []  # storing the best ant at each epoch
        self.current_best_X, self.current_best_Y = [], []
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y
        self.best_x, self.best_y = None, None
        self.branching_factor = []
        if self.save_file is not None:
            self._save_params()

    def _save_params(self):
        p = {
            "size" : self.n_dim,
            "size_pop": self.size_pop,
            "max_iter": self.max_iter,
            "alpha": self.alpha,
            "beta": self.beta,
            "peromone_decay": self.rho,
            "min_tau": self.min_tau,
            # "bias": self.prob_bias,
            "branching_factor": self.lamb,
            "checkpoint_res": self._checkpoint_res
        }
        with open(self.save_file, "wb") as f:
            f.write((str(p) + "\n").encode())

    def _branching_factor(self) -> float:
        # assert (0 <= lambda) and (lambda <=1)
        tau_min = self.Tau.min(axis=1).reshape((-1, 1))
        tau_max = self.Tau.max(axis=1).reshape((-1, 1))
        return np.where(self.Tau >= tau_min + self.lamb * (tau_max-tau_min))[0].size

    def _delta_tau(self) -> np.ndarray:
        """Calculate the update rule."""
        delta_tau = np.zeros((self.n_dim, self.n_dim))
        for j in range(self.size_pop):  # per ant
            # add 1/(path len) to each edge
            for k in range(self.n_dim - 1):
                n1, n2 = self.Table[j, k], self.Table[j, k + 1]
                delta_tau[n1, n2] += 1 / self.y[j]
            n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]
            delta_tau[n1, n2] += 1 / self.y[j]
        return delta_tau


    def _phero_update(self) -> None:
        """
        Take an update step
        tau_{ij} <- rho * tau_{ij} + delta tau_{ij}
        """
        self.Tau = (1 - self.rho) * self.Tau + self.rho * self._delta_tau()

    def _get_candiates(self, taboo_set) -> list:
        """Get the availible nodes that are not in the taboo list."""
        return list(set(range(self.n_dim)) - set(taboo_set))

    def _ant_search(self, j) -> None:
        """Find a path for a single path."""
        self.Table[j, 0] = 0  # start at node 0
        for k in range(self.n_dim - 1):
            # get viable
            allow_list = self._get_candiates(set(self.Table[j, :k + 1]))
            # roulette selection
            prob = self.prob_matrix[self.Table[j, k], allow_list]
            prob = prob / prob.sum()
            next_point = np.random.choice(allow_list, size=1, p=prob)[0]
            self.Table[j, k + 1] = next_point

    def _prob_rule(self):
        return (self.Tau ** self.alpha) * (self.prob_bias ** self.beta)

    def _prob_rule_node(self, node, taboo_list) -> float:
        probs = np.zeros(self.n_dim)
        allow_list = self._get_candiates(taboo_set=taboo_list)
        probs[allow_list] = self.prob_matrix[node][allow_list]
        probs /= probs.sum()
        return probs[node]

    def _get_best(self):
        index_best = self.y.argmin()
        x_best, y_best = self.Table[index_best, :].copy(), self.y[index_best].copy()
        self.generation_best_X.append(x_best)
        self.generation_best_Y.append(y_best)
        best_generation = np.array(self.generation_best_Y).argmin()
        self.current_best_X.append(self.generation_best_X[best_generation])
        self.current_best_Y.append(self.generation_best_Y[best_generation])
        if self.metric_branching_factor:
            self.branching_factor.append(self._branching_factor())
        return x_best, y_best

    def set_metrics(self, metrics):
        if "best" in metrics:
            self.metric_best = True
        if "branching_factor" in metrics:
            self.metric_branching_factor = True

    def _checkpoint(self) -> None:
        """Pickles self to disk."""
        with open(self._checkpoint_file, "wb") as f:
            pickle.dump(self, f)

    def _save(self) -> None:
        """Saves learned pheromone table to disk."""
        with open(self.save_file, "ab") as f:
            f.write(self.Tau.astype(np.float64).tobytes())

    def _load(self, filename) -> None:
        """Loads learned pheromone table from disk."""
        with open(filename, "rb") as f:
            params = ast.literal_eval(f.readline().decode())
            tau_table = np.frombuffer(f.read(), dtype=np.float64)
            tau_table = tau_table.reshape((-1, params["size"], params["size"]))
        return params, tau_table

    def _restore(self) -> None:
        pass

    def run(self,
            max_iter=None,
            metric= ["best"]):
        if self.func is None or self.distance_matrix is None:
            raise ValueError("func and distance_matrix must be set to run ACA")

        self.set_metrics(metric)
        self.metrics = metric
        """Driver method for finding the opt to TSP problem."""
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.iteration = i
            self.prob_matrix = self._prob_rule()
            for j in range(self.size_pop):
                self._ant_search(j)
            # get score of ants via given opt funciton
            self.y = np.array([self.func(i) for i in self.Table])
            self._get_best()

            self._phero_update()

            if self.save_file is not None and self.iteration % self._checkpoint_res == 0:
                self._save()
            if self._checkpoint_file is not None and self.iteration % self._checkpoint_res == 0:
                self._checkpoint()

        self.best_x = self.current_best_X[-1]
        self.best_y = self.current_best_Y[-1]
        return self.best_x, self.best_y

if __name__ == "__main__":
    size = 100
    runs = 1
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))
    # distance_matrix[np.where(distance_matrix == 0)] = 1e13

    def cal_total_distance(routine):
        size = len(routine)
        return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                    for i in range(size)])


    print("Running ACA")
    ACA_runs = []

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = ACA_TSP(func=cal_total_distance,
                      distance_matrix=distance_matrix,
                      params={"max_iter": iterations,
                              "save_file": save_file,
                              "checkpoint_res": 1,
                              "rho": 0.5,
                              "alpha": 1,
                              "beta": 1,
                              "pop_size": 10,
                              "bias": "inv_weight"
                              })
        skaco_sol, skaco_cost = aca.run(metric = ["branching_factor"])
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    # params, tau_table = aca._load("ACO_run_0.txt")
    # print(tau_table)

    G = nx.from_numpy_array(distance_matrix, create_using=nx.DiGraph)
    approx = nx.approximation.simulated_annealing_tsp(G, "greedy", source=0)


    print(cal_total_distance(approx))

    pass
