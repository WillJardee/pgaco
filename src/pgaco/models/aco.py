#!/usr/bin/env python3
"""
Ant Colony Optimization (ACO) implementation for solving TSP problems.

This module provides classes and functions to implement the ACO algorithm,
particularly focused on solving the Traveling Salesman Problem (TSP).

Classes:
    ACO: Main class for the Ant Colony Optimization algorithm.

The ACO algorithm simulates the behavior of ants to find optimal paths
in a graph, which can be applied to various shortest problems.
"""

import warnings
import inspect
from typing import Callable, Iterable

import numpy as np

from pgaco.models import ACOBase
from pgaco.models import post_init_decorator

def path_len(distance_matrix: np.ndarray, path) -> float:
    length = len(path)
    cost = 0
    for i in range(length):
        cost += distance_matrix[path[i%length]][path[(i+1)%length]]
    return float(cost)

class ACO(ACOBase):
    """Simple, default ACO solution to TSP problem."""
    @post_init_decorator
    def __init__(self,
                 distance_matrix: np.ndarray,
                 func: Callable[[np.ndarray, Iterable], float] = path_len,
                 *,
                 size_pop: int = 10,
                 alpha: float = 1,
                 beta: float = 2,
                 evap_rate: float = 0.1,
                 minmax: bool = True,
                 replay_size: int = 1,
                 replay_rule: str = "elite",
                 **kwargs,
                 ) -> None:
        """
        Builder for base ACO class.

        Parameters
        ----------
            distance_matrix : (np.ndarray)
                Weighted adjacency matrix of the given TSP space.
            func : function, default : path_len
                The function to minimize.
            size_pop : int, default 10
                Number of ants to search with at each generation
            alpha : float, default 1
                Exponential weight to apply to the learned heuristic information.
            beta : float, default 2
                Exponential to apply to the prior heuristic information.
            evap_rate : float, default 0.1
                Ratio to remove at each time step.
            minmax : bool, default True
                Whether to use the adaptive minmax rule
            replay_size : int, default 1
                Size of replay buffer.
            replay_rule : str, elite
                Type of replay rule to use. Options: elite, evict, none
                elite: keep the best `replay_size` ants
                evict: keep the most recent `replay_size` ants (notice that if `replay_size` < `size_pop` the extra ants will still be evaluated.).
                none: the ants at a generation will reinforce. This is equivalent to `replay_size` = `size_pop` with `replay_rule` = "evict".
        """
        self._name_ = "ACO"
        super().__init__(**kwargs)

        self.distance_matrix = distance_matrix.astype(np.float64) # cost matrix
        self.func = func

        self._size_pop = size_pop
        self._alpha = alpha
        self._beta = beta
        self._evap_rate = evap_rate
        self._replay_size = replay_size
        self._replay_rule = replay_rule
        self._minmax_adaptive = minmax
        self._iteration         =   0

        self._initialize_workspace()
        self._prob_rule_update()


    def _initialize_workspace(self) -> None:
        self._min_dist  =   self.distance_matrix[np.where(self.distance_matrix > 0)].min()
        self._max_dist  =   self.distance_matrix.max()
        self._dim       =   distance_matrix.shape[0]
        self._max_tau = 1/(self._dim * self._min_dist) # max value; helps with runaway behavior
        self._min_tau = self._max_tau/(2 * self._dim) # min value; helps erogtic behavior and NaNs
        # self._min = False if (self._min_tau == -1 and not self._minmax_adaptive) else True
        # min always has to be taken or else we get an undefined point
        if self._min_tau == -1:
            self._min_tau = self._min_dist * 1e-7
        self._min = True
        self._max = False if (self._max_tau == -1 and not self._minmax_adaptive) else True
        # building workspace
        self.distance_matrix += 1e-10 * np.eye(self._dim) # Helps with NaN and stability of some methods

        self._prob_bias = 1 / (self.distance_matrix) # bias value (1/len)
        self._prob_bias[np.where(self._prob_bias == np.inf)] = 0
        self._heuristic_table = self._prob_bias.copy()
        num_list = np.arange(self._dim)
        self._replay_buffer = np.array([self._rng.permutation(num_list) for _ in range(self._replay_size)])
        self._replay_buffer_fit = np.array([self.func(self.distance_matrix, i) for i in self._replay_buffer])
        self.current_best_Y, self.current_best_X = self._get_best(n=1)
        self.generation_best_Y, self.generation_best_X = [self.current_best_Y], [self.current_best_X]  # storing the best ant at each epoch
        # self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y
        self.best_x, self.best_y = None, None


    def _validate_params(self) -> None:
        super()._validate_params()
        assert callable(self.func)
        assert isinstance(self._size_pop, int) and self._between(self._size_pop, lower=1, inclusive=True)
        assert isinstance(self._alpha, float) and self._between(self._alpha, lower=0, inclusive=True)
        assert isinstance(self._beta, float) and self._between(self._beta, lower=0, inclusive=True)
        assert isinstance(self._evap_rate, float) and self._between(self._evap_rate, lower=0, upper=1, inclusive=True)
        assert isinstance(self._replay_size, int) and self._between(self._size_pop, lower=1, inclusive=True)
        assert isinstance(self._replay_rule, str) and self._replay_rule in ["elite", "evict", "none"]
        assert isinstance(self._minmax, bool)


    def _minmax(self) -> None:
        if self._max:
            if self._minmax_adaptive:
                self._max_tau = 1/(self._evap_rate * self.current_best_Y)
            self._heuristic_table[np.where(self._heuristic_table > self._max_tau)] = self._max_tau
        if self._min:
            if self._minmax_adaptive:
                self._min_tau = self._max_tau / (2 * self._dim)
            self._heuristic_table[np.where(self._heuristic_table < self._min_tau)] = self._min_tau


    def _gradient(self, solution, cost) -> np.ndarray:
        """Calculate the gradient for a single example."""
        sol_len = len(solution)
        # add 1/(path len) to each edge
        grad = np.zeros(self._heuristic_table.shape)
        for k in range(sol_len):
            n1, n2 = solution[(k)%sol_len], solution[(k+1)%sol_len]
            grad[n1, n2] += 1 / cost
            # grad[n2, n1] += 1 / cost
        return grad

    def _gradient_update(self) -> None:
        """Take an gradient step."""
        tot_grad = np.zeros(self._heuristic_table.shape)
        for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
            tot_grad += self._gradient(solution, cost)
        tot_grad = tot_grad/self._replay_size

        self._heuristic_table = (1 - self._evap_rate) * self._heuristic_table + self._evap_rate * tot_grad
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
            next_point = self._rng.choice(allow_list, p=prob) # roulette selection
            solution.append(next_point)
        return np.array(solution, dtype=int)

    def _prob_rule_update(self) -> None:
        """Update the probability matrix."""
        self._prob_matrix = (self._heuristic_table ** self._alpha) * (self._prob_bias ** self._beta)

    def _prob_rule_node(self, node: int, taboo_list: set[int] | list[int]) -> float:
        """Return the probability of an action given a state and history."""
        probs = np.zeros(self._dim)
        allow_list = self._get_candiates(taboo_set=taboo_list)
        probs[allow_list] = self._prob_matrix[node][allow_list]
        probs /= probs.sum()
        return probs[node]

    def _elite_replay_rule(self, new_fitness, new_solutions) -> None:
        full_buffer = np.concatenate([new_solutions, self._replay_buffer])
        full_fitness = np.concatenate([new_fitness, self._replay_buffer_fit])
        keep_indices = np.argpartition(full_fitness, kth=self._replay_size)[:self._replay_size]
        self._replay_buffer = full_buffer[keep_indices]
        self._replay_buffer_fit = full_fitness[keep_indices]

    def _evict_replay_rule(self, new_fitness, new_solutions) -> None:
        full_buffer = np.concatenate([new_solutions, self._replay_buffer])
        full_fitness = np.concatenate([new_fitness, self._replay_buffer_fit])
        self._replay_buffer = full_buffer[:self._replay_size]
        self._replay_buffer_fit = full_fitness[:self._replay_size]

    def _add_replay_buffer(self, new_fitness, new_solutions) -> None:
        """Add the provided list to the replay buffer, prefering new values when relevant."""
        match self._replay_rule:
            case "none":
                self._replay_buffer = new_solutions
                self._replay_buffer_fit = new_fitness
            case "elite":
                self._elite_replay_rule(new_fitness, new_solutions)
            case "evict":
                self._evict_replay_rule

    def get_solution(self, start: int | None = None, seed: int | None =  None) -> tuple[float, np.ndarray]:
        if start is None:
            rng = np.random.default_rng(seed = seed or self._seed)
            start = rng.integers(self._dim)
        assert start >= 0
        path = [start]
        for _ in range(self._dim - 1):
            allow_list = self._get_candiates(path)
            next_point = allow_list[self._heuristic_table[path[-1], allow_list].argmax()]
            path.append(next_point)
        return self.func(self.distance_matrix, path), np.array(path)

    def _get_best(self, n: int = 1, sort: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return the n best values in the replay buffer."""
        assert isinstance(sort, bool)

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


    def take_step(self, steps: int = 1) -> tuple[float, np.ndarray]:
        """Take [steps; default=1] steps of the search algorithm."""
        self.generation_policy_score = []
        for _ in range(steps):
            self._iteration += 1
            self._prob_rule_update()

            # Generate solutions
            gen_solutions = [self._single_solution() for _ in range(self._size_pop)]
            gen_fitness = np.array([self.func(self.distance_matrix, i) for i in gen_solutions])
            self._add_replay_buffer(gen_fitness, gen_solutions)

            self._gradient_update()

            # Get best solution and save it
            y, x = self._get_best(n=1)
            if y <= self.current_best_Y:
                self.current_best_Y, self.current_best_X = float(y), x
            self.generation_best_Y.append(y)
            y_policy, _ = self.get_solution()
            self.generation_policy_score.append(y_policy)

            # Save check
            if self._iteration % self._checkpoint_res == 0:
                if self._savefile is not None:
                    self._save()
                if self._checkpointfile is not None:
                    self._checkpoint()

        return float(self.current_best_Y), self.current_best_X


    def run(self, max_iter: int = 100):
        """Run through solving the TSP."""
        if self.func is None or self.distance_matrix is None:
            raise ValueError(f"func and distance_matrix must be set to run {self._name_}")

        if self._iteration > 0:
            warnings.warn(f"The model has already make {self._iteration}, taking another {max_iter} steps. Recreate the class to reset the search",
                          stacklevel=2)

        return self.take_step(steps=max_iter)

def run_model1(distance_matrix, seed):
    aco = ACO(distance_matrix,
                  beta          = 0,
                  size_pop      = 2,
                  max_iter      = max_iter,
                  replay_rule   = "global_best",
                  seed          = seed)
    aco.run()
    return aco.generation_best_Y, aco.generation_policy_score, "MINMAX " + aco._name_

def run_model2(distance_matrix, seed):
    aco = ACO(distance_matrix,
                  evap_rate     = 0.1,
                  alpha         = 1,
                  beta          = 2,
                  size_pop      = 2,
                  max_iter      = max_iter,
                  replay_rule   = "global_best",
                  seed          = seed)
    aco.run()
    return aco.generation_best_Y, aco.generation_policy_score, "MINMAX " + aco._name_



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pgaco.utils import get_graph, plot, parallel_runs
    size = 20
    runs = 5
    max_iter = 1500
    distance_matrix = get_graph(size)

    print("running MMACO (beta = 0)")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model1, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="cyan", label=aco_name)
    plot(aco_policy_runs, color="blue", label=aco_name + " policy, beta=0")

    print("running MMACO (beta = 2)")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model2, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="green", label=aco_name)
    plot(aco_policy_runs, color="lime", label=aco_name + " policy, beta=2")

    plt.legend()
    plt.tight_layout()
    plt.show()

    pass