#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/4/2
# @Author  : github.com/willjardee

# pylint: disable=line-too-long

"""
Change Log:
- renamed parameters
- made parameters kwargs
- removed branching factor
"""
from typing import Tuple
import pickle
import ast
import warnings

import numpy as np

class ACO_TSP:
    """Simple, default ACO solution to TSP problem."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 **kwargs) -> None:
        """
        Builder for base ACO class.

        Args:
            distance_matrix (np.ndarray) = weighted adjacency matrix of the given TSP space.
            opt_func (function) = function to optimize wrt. default: path cost
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
        self._name_ = "ACO"

        self.distance_matrix = distance_matrix.astype(np.float64) # cost matrix

        self._min_dist  =   self.distance_matrix[np.where(self.distance_matrix > 0)].min()
        self._max_dist  =   self.distance_matrix.max()
        self._dim       =   distance_matrix.shape[0]


        # Setting parameters
        self.allowed_params = {"size_pop", "max_iter", "alpha", "beta",
                               "evap_rate", "min_tau", "max_tau", "minmax",
                               "bias_func", "save_file", "checkpoint_file",
                               "checkpoint_res", "replay_size", "seed"}
        for key in kwargs:
            if key not in self.allowed_params:
                warnings.warn(f"Parameter '{key}' is not recognized and will be ignored.")

        self.func               =   kwargs.get("opt_func", self._path_len)
        self._size_pop          =   kwargs.get("size_pop", 10)
        self._max_iter          =   kwargs.get("max_iter", 100)
        self._alpha             =   kwargs.get("alpha", 1)
        self._beta              =   kwargs.get("beta", 2)
        self._evap_rate         =   kwargs.get("evap_rate", 0.1)
        self._replay_size       =   kwargs.get("replay_size", self._size_pop) # assumes no replay buffer
        self._replay = False if self._replay_size == -1 else True
        if not self._replay: self._replay_size = self._size_pop
        self._min_tau           =   kwargs.get("min_tau", 1/self._max_dist) # min value; helps erogtic behavior and NaNs
        self._max_tau           =   kwargs.get("max_tau", self._max_dist) # max value; helps with runaway behavior
        self._minmax_adaptive   =   kwargs.get("minmax", True)
        # self._min = False if (self._min_tau == -1 and not self._minmax_adaptive) else True
        # min always has to be taken or else we get an undefined point
        if self._min_tau == -1: self._min_tau = self._min_dist * 1e-7
        self._min = True
        self._max = False if (self._max_tau == -1 and not self._minmax_adaptive) else True
        self._bias_func         =   kwargs.get("bias_func", "inv_weight") # Uniform and inv_weight
        self.save_file          =   kwargs.get("save_file", None) # relative path
        self.checkpoint_file    =   kwargs.get("checkpoint_file", None)
        self._checkpoint_res    =   kwargs.get("checkpoint_res", 10)
        self._seed              =   kwargs.get("seed", None)

        self._rng               =   np.random.default_rng(seed=self._seed)
        self._iteration         =   0

        # building workspace
        self._heuristic_table = np.ones((self._dim, self._dim)) * (self._max_tau if self._max else 1) # This is where the parameters of the learned policy are stored
        self.distance_matrix += 1e-10 * np.eye(self._dim) # Helps with NaN and stability of some methods

        # set probability bias
        match self._bias_func.lower():
            case "uniform": # default is uniform
                self._prob_bias = np.ones((self.distance_matrix.shape))
            case "inv_weight": # inverse distance
                self._prob_bias = 1 / (self.distance_matrix) # bias value (1/len)
                self._prob_bias[np.where(self._prob_bias == np.inf)] = 0
            case _:
                raise ValueError(f"Invalid bias value of {self._bias_func}")

        self._prob_rule_update()

        num_list = np.arange(self._dim)
        self._replay_buffer = np.array([self._rng.permutation(num_list) for _ in range(self._replay_size)])
        self._replay_buffer_fit = np.array([self.func(i) for i in self._replay_buffer])
        self.generation_best_X, self.generation_best_Y = [], []  # storing the best ant at each epoch
        self.current_best_Y, self.current_best_X = self._replay_buffer_fit[0], np.array([])
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y
        self.best_x, self.best_y = None, None
        if self.save_file is not None: self._save_params()

    def _passkwargs(self, **kwargs):
        passkwargs = kwargs.copy()
        for key in kwargs:
            if key in self.allowed_params:
                passkwargs.pop(key)
        return passkwargs

    def _save_params(self, filename: str = ""):

        """Save the parameters as a header to a file."""
        p = {
            "size" : self._dim,
            "size_pop": self._size_pop,
            "max_iter": self._max_iter,
            "alpha": self._alpha,
            "beta": self._beta,
            "peromone_decay": self._evap_rate,
            "min_tau": self._min_tau,
            "bias": self._bias_func,
            "checkpoint_res": self._checkpoint_res
            }
        if filename == "": filename = self.save_file
        with open(filename, "wb") as f:
            f.write((str(p) + "\n").encode())

    def _checkpoint(self) -> None:
        """Pickles self to disk."""
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(self, f)

    def _save(self) -> None:
        """Saves learned pheromone table to disk."""
        with open(self.save_file, "ab") as f:
            f.write(self._heuristic_table.astype(np.float64).tobytes())

    def _load(self, filename):
        """Loads learned pheromone table from disk."""
        with open(filename, "rb") as f:
            params = ast.literal_eval(f.readline().decode())
            tau_table = np.frombuffer(f.read(), dtype=np.float64)
            tau_table = tau_table.reshape((-1, params["size"], params["size"]))
        return params, tau_table

    def _restore(self) -> None:
        pass

    def _path_len(self, path) -> float:
        length = len(path)
        cost = 0
        for i in range(length + 1):
            cost += self.distance_matrix[path[i%length]][path[(i+1)%length]]
        return float(cost)

    def _gradient(self) -> np.ndarray:
        """Calculate the update rule."""
        delta = np.zeros(self._heuristic_table.shape)
        for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
            sol_len = len(solution)
            # add 1/(path len) to each edge
            for k in range(sol_len + 1):
                n1, n2 = solution[(k)%sol_len], solution[(k+1)%sol_len]
                delta[n1, n2] += 1 / cost
        return delta

    def _minmax(self) -> None:
        if self._max:
            if self._minmax_adaptive:
                self._max_tau = 1/(self._evap_rate * self.current_best_Y)
            self._heuristic_table[np.where(self._heuristic_table > self._max_tau)] = self._max_tau
        if self._min:
            if self._minmax_adaptive:
                self._min_tau = self._max_tau / (2 * self._dim)
            self._heuristic_table[np.where(self._heuristic_table < self._min_tau)] = self._min_tau


    def _gradient_update(self) -> None:
        """Take an gradient step"""
        self._heuristic_table = (1 - self._evap_rate) * self._heuristic_table + self._evap_rate * self._gradient()
        self._minmax()


    def _get_candiates(self, taboo_set: set[int] | list[int]) -> list:
        """Get the availible nodes that are not in the taboo list."""
        return list(set(range(self._dim)) - set(taboo_set))

    def _single_solution(self) -> np.ndarray:
        """Find a path for a single path."""
        solution = [self._rng.integers(self._dim)]
        for k in range(self._dim - 1):
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = self._prob_matrix[solution[k], allow_list]
            prob = prob / prob.sum()
            next_point = self._rng.choice(allow_list, size=1, p=prob)[0] # roulette selection
            solution.append(next_point)
        return np.array(solution, dtype=int)

    def _prob_rule_update(self) -> None:
        """Updates the probability matrix"""
        self._prob_matrix = (self._heuristic_table ** self._alpha) * (self._prob_bias ** self._beta)

    def _prob_rule_node(self, node: int, taboo_list: set[int] | list[int]) -> float:
        """Returns the probability of an action given a state and history"""
        probs = np.zeros(self._dim)
        allow_list = self._get_candiates(taboo_set=taboo_list)
        probs[allow_list] = self._prob_matrix[node][allow_list]
        probs /= probs.sum()
        return probs[node]

    def _add_replay_buffer(self, new_fitness, new_solutions, sort: bool = True) -> None:
        """Add the provided list to the replay buffer, prefering new values when relevant"""
        if not self._replay: # if we are not doing replay, just return the values
            self._replay_buffer = new_solutions
            self._replay_buffer_fit = new_fitness
            return

        full_buffer = np.concatenate([new_solutions, self._replay_buffer])
        full_fitness = np.concatenate([new_fitness, self._replay_buffer_fit])
        if sort:
            keep_indices = np.argsort(full_fitness)[:self._replay_size]
        else:
            keep_indices = np.argpartition(full_fitness, kth=self._replay_size)[:self._replay_size]
        self._replay_buffer = full_buffer[keep_indices]
        self._replay_buffer_fit = full_fitness[keep_indices]


    def _get_best(self, n: int = 1, sort: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the n best values in the replay buffer"""
        assert type(sort) == bool

        if n == 1:
            index = self._replay_buffer_fit.argmin()
            return np.array(self._replay_buffer_fit[index]), np.array(self._replay_buffer[index])
        elif n > 1 and n <= self._dim:
            if sort:
                index = np.argsort(self._replay_buffer_fit)[:n]
            else:
                index = np.argpartition(self._replay_buffer_fit, kth=n)[:n]
            return self._replay_buffer_fit[index], self._replay_buffer[index]
        else:
            raise ValueError("self._get_best only supports up to the length of the replay_buffer")


    def take_step(self, steps=1) -> Tuple[float, np.ndarray]:
        """Take [steps; default=1] steps of the search algorithm"""
        for i in range(steps):
            self._iteration += i
            self._prob_rule_update()

            # Generate solutions
            gen_solutions = [self._single_solution() for _ in range(self._size_pop)]
            gen_fitness = np.array([self.func(i) for i in gen_solutions])
            self._add_replay_buffer(gen_fitness, gen_solutions)

            self._gradient_update()

            # Get best solution and save it
            y, x = self._get_best(n=1)
            if y <= self.current_best_Y: self.current_best_Y, self.current_best_X = float(y), x
            self.generation_best_Y.append(y)

            # Save check
            if self._iteration % self._checkpoint_res == 0:
                if self.save_file is not None: self._save()
                if self.checkpoint_file is not None: self._checkpoint()

        return float(self.current_best_Y), self.current_best_X


    def run(self, max_iter=None):
        """Runs through solving the TSP."""
        if self.func is None or self.distance_matrix is None:
            raise ValueError(f"func and distance_matrix must be set to run {self._name_}")

        num_iter = max_iter or self._max_iter

        if self._iteration > 0:
            warnings.warn(f"The model has already make {self._iteration}, taking another {num_iter} steps. Recreate the class to reset the search")

        return self.take_step(steps=num_iter)

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    size = 100
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))

    print("Running ACA")
    ACA_runs = []
    aca = ACO_TSP(distance_matrix,
                  max_iter = iterations,
                  replay_size = -1)

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = ACO_TSP(distance_matrix,
                      max_iter = iterations,
                      save_file = save_file,
                      replay_size = -1)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
