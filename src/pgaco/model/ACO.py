#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/4/2
# @Author  : github.com/willjardee

# pylint: disable=line-too-long

import pickle
import ast
import warnings

import numpy as np

def shortest_path(adj_mat: np.ndarray, path: np.ndarray):
    length = len(path)
    cost = 0
    for i in range(length + 1):
        cost += adj_mat[[path[i%length], path[(i+1)%length]]]
    return cost

class ACO_TSP:
    """Simple, default ACO solution to TSP problem."""


    def __init__(self,
                 distance_matrix: np.ndarray,
                 func = shortest_path,
                 **kwargs) -> None:
        """
        Builder for base ACA class.

        Args:
            distance_matrix (np.ndarray) = weighted adjacency matrix of the given TSP space.
            func (function) = function to optimize wrt. default: path cost
            size_pop (int) = Number of ants to search with at each generation. default: 10
            max_iter (int) = Max number of iteration to allow. default: 100
            alpha (float) = Exponential to apply to the learned heuristic information. default: 1
            beta (float) = Exponential to apply to the prior heuristic information. default: 2
            evap_rate (float) = Ratio to remove at each time step. default: 0.1
            replay_size (int) = Size of replay buffer. A value of -1 disables the replay buffer (replay_size = size_pop). If replay_size < size_pop only the top relay_size ants will lay pheromone. default: -1
            min_tau (float) = Min value allowed for learned heuristic values. default: 0.001 * min(distance_matrix)
            bias_func (str) = Table to be used an initial heuristic information for the problem. Takes: 'uniform' and 'inv_weight'. default: inv_weight
            save_file (str) = Relative path to save the final model to. default: None
            checkpoint_file (str) = Relative path to save checkpoint file to. default: None
            checkpoint_res (int) = Number of steps between saving the model. default: 10
        """

        self.distance_matrix = distance_matrix # cost matrix
        self.func = func # optimization function

        self._min_dist  =   self.distance_matrix.min()
        self._max_dist  =   self.distance_matrix.max()
        self._dim       =   distance_matrix.shape[0]

        self.allowed_params = {"size_pop", "max_iter", "alpha", "beta",
                               "evap_rate", "min_tau", "bias_fucn",
                               "save_file", "checkpoint_file",
                               "checkpoint_res"}
        for key in kwargs:
            if key not in self.allowed_params:
                warnings.warn(f"Parameter '{key}' is not recognized and will be ignored.")

        self._size_pop          =   kwargs.get("size_pop", 10)
        self._max_iter          =   kwargs.get("max_iter", 100)
        self._alpha             =   kwargs.get("alpha", 1)
        self._beta              =   kwargs.get("beta", 2)
        self._evap_rate         =   kwargs.get("evap_rate", 0.1)
        self._replay_size       =   kwargs.get("replay_size", -1) # assumes no replay buffer
        self._min_tau           =   kwargs.get("min_tau", 0.001 * self._min_dist) # min value; helps erogtic behavior and NaNs
        self._bias_func         =   kwargs.get("bias_func", "inv_weight") # Uniform and inv_weight
        self.save_file          =   kwargs.get("save_file", None) # relative path
        self.checkpoint_file    =   kwargs.get("checkpoint_file", None)
        self._checkpoint_res    =   kwargs.get("checkpoint_res", 10)

        if self.distance_matrix is not None:
            self._build_workspace()
        self._name_ = "ACO - revised"

    def _build_workspace(self):
        # building workspace
        self._heuristic_table = np.ones((self._dim, self._dim))
        self.distance_matrix += 1e-10 * np.eye(self._dim)

        # set probability bias
        match self._bias_func.lower():
            case "uniform": # default is uniform
                self.prob_bias = np.ones((self.distance_matrix.shape))
            case "inv_weight": # inverse distance
                self.prob_bias = 1 / (self.distance_matrix) # bias value (1/len)
                self.prob_bias[np.where(self.prob_bias == np.inf)] = 0
            case _:
                raise ValueError(f"Invalid bias value of {self.prob_bias}")

        self.Table = np.zeros((self._size_pop, self._dim)).astype(int)  # set of ant solutions; row - ant in order of visit
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

    def _get_best_legacy(self, n: int = 1, sort: bool = False):
        """
        Returns the n best
        Args:
            n (int): (default = 1) number to return
            sort (bool): (defaults = False) Whether to return a sorted version of the list
        Returns:
            sorted list of the sort x_best, y_best
        """
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

    def _get_best(self,
                  n: int = 1,
                  sort: bool = False):
        """
        Returns the n best
        Args:
            n (int): (default = 1) number to return
            sort (bool): (defaults = False) Whether to return a sorted version of the list
        Returns:
            sorted list of (cost, path)
        """
        assert type(sort) == bool

        grab_list = np.array([[self.y[i], self.Table[i]] for i in range(len(self.y))])
        smallest_indices = np.argpartition([x[0] for x in grab_list], n)[:n]
        shortest_list = [grab_list[i] for i in smallest_indices]

        if not sort:
            return shortest_list
        else:
            sorted_indices = np.argsort([x[0] for x in shortest_list])
            sorted_list = [shortest_list[i] for i in sorted_indices]
            return sorted_list

        self.generation_best_X.append(x_best)
        self.generation_best_Y.append(y_best)
        best_generation = np.array(self.generation_best_Y).argmin()
        self.current_best_X.append(self.generation_best_X[best_generation])
        self.current_best_Y.append(self.generation_best_Y[best_generation])
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
