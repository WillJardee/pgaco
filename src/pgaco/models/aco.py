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
from typing import Callable, Iterable

from scipy.sparse import dok_matrix
import numpy as np

from pgaco.models import ACOBase, path_len

class ACO(ACOBase):
    """Simple, default ACO solution to TSP problem."""
    def __init__(self,
                 distance_matrix: np.ndarray,
                 func: Callable[[np.ndarray, Iterable], float] = path_len,
                 *,
                 size_pop: int = 10,
                 alpha: float = 1,
                 beta: float = 2,
                 evap_rate: float = 0.1,
                 minmax: bool = True,
                 replay_size: int = 20,
                 update_size: int | None = None,
                 replay_rule: str = "elite",
                 slim: bool = True,
                 softmax: bool = False,
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
                Size of replay buffer to keep.
            update_size : int | None, default None
                Number of samples from the replay buffer to use in the update step.
            replay_rule : str, elite
                Type of replay rule to use. Options: elite, evict, none
                elite: keep the best `replay_size` ants
                evict: keep the most recent `replay_size` ants (notice that if `replay_size` < `size_pop` the extra ants will still be evaluated.).
                none: the ants at a generation will reinforce. This is equivalent to `replay_size` = `size_pop` with `replay_rule` = "evict".
            slim : bool, True
                Whether to do all calculation for introspection.
                Disabling leaves: generation_best_Y, generation_policy_score, generation_best_X empty
        """
        super().__init__(**kwargs)

        self.distance_matrix = distance_matrix.astype(np.float64) # cost matrix
        self.func = func

        self._size_pop = size_pop
        self._alpha = alpha
        self._beta = beta
        self._evap_rate = evap_rate
        self._replay_rule = replay_rule
        if self._replay_rule == "none":
            self._replay_size = size_pop
        elif self._replay_rule == "global_best":
            self._replay_rule = "elite"
            self._replay_size = 1
        else:
            self._replay_size = replay_size
        if update_size is not None:
            self._update_size = update_size
        else:
            self._update_size = self._replay_size
        self._minmax_adaptive = minmax
        self.slim = slim
        self._softmax = softmax

        self._initialize_workspace()
        self._prob_rule_update()
        self._name_ = "ACO"


    def _initialize_workspace(self) -> None:
        self._min_dist  =   self.distance_matrix[np.where(self.distance_matrix > 0)].min()
        self._max_dist  =   self.distance_matrix.max()
        self._dim       =   self.distance_matrix.shape[0]
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

        if self._softmax:
            self._prob_rule_update = self._prob_rule_update_softmax
        self._prob_bias = np.matrix(1 / (self.distance_matrix)) # bias value (1/len)
        self._prob_bias[np.where(self._prob_bias == np.inf)] = 0
        self._heuristic_table = self._prob_bias.copy()
        num_list = np.arange(self._dim)
        self._replay_buffer = np.array([self._rng.permutation(num_list) for _ in range(self._replay_size)])
        self._replay_buffer_probs = np.array([np.ones(self._dim) for _ in range(self._replay_size)])
        self._replay_buffer_fit = np.array([self.func(self.distance_matrix, i) for i in self._replay_buffer])
        self._replay_buffer_grads = np.array([dok_matrix((self._dim, self._dim)) for _ in range(self._replay_size)])
        self.current_best_Y, self.current_best_X = self._get_best(n=1)
        self.generation_best_Y, self.generation_best_X = [self.current_best_Y], [self.current_best_X]  # storing the best ant at each epoch
        # self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y
        self.best_x, self.best_y = None, None

    """The following is collection of parameter validation rules."""
    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, func):
        assert callable(func)
        self._func = func

    @property
    def _size_pop(self):
        return self.__size_pop

    @_size_pop.setter
    def _size_pop(self, size_pop):
        assert self._between(size_pop, lower=0)
        self.__size_pop = int(size_pop)

    # @property
    # def _alpha(self):
    #     return self.__alpha
    #
    # @_alpha.setter
    # def _alpha(self, alpha):
    #     assert self._between(alpha, lower=0, inclusive=True)
    #     self.__alpha = float(alpha)

    @property
    def _beta(self):
        return self.__beta

    @_beta.setter
    def _beta(self, beta):
        assert self._between(beta, lower=0, inclusive=True)
        self.__beta = float(beta)

    @property
    def _evap_rate(self):
        return self.__evap_rate

    @_evap_rate.setter
    def _evap_rate(self, evap_rate):
        assert self._between(evap_rate, lower=0, inclusive=True)
        self.__evap_rate = float(evap_rate)

    @property
    def _replay_size(self):
        return self.__replay_size

    @_replay_size.setter
    def _replay_size(self, replay_size):
        assert self._between(replay_size, lower=1, inclusive=True)
        self.__replay_size = int(replay_size)

    @property
    def _update_size(self):
        return self.__update_size

    @_replay_size.setter
    def _update_size(self, update_size):
        if not hasattr(self, '_replay_size'): NameError('_replay_size must be set before calling _update_size')
        assert self._replay_size >= update_size, "replay_size  must be at least as large as update_size"
        self.__update_size = int(update_size)


    @property
    def _replay_rule(self):
        return self.__replay_rule

    @_replay_rule.setter
    def _replay_rule(self, replay_rule):
        valid_replay = ["global_best", "elite", "evict", "none"]
        assert replay_rule in valid_replay, f"replay_rule must be one of {valid_replay}"
        self.__replay_rule = replay_rule

    @property
    def _minmax_adaptive(self):
        return self.__minmax_adaptive

    @_minmax_adaptive.setter
    def _minmax_adaptive(self, minmax_adaptive):
        assert isinstance(minmax_adaptive, bool)
        self.__minmax_adaptive = minmax_adaptive

    def _minmax(self) -> None:
        """Apply the adaptive MinMax rule."""
        if self._max:
            if self._minmax_adaptive:
                self._max_tau = 1/(self._evap_rate * self.current_best_Y)
            self._heuristic_table[np.where(self._heuristic_table > self._max_tau)] = self._max_tau
        if self._min:
            if self._minmax_adaptive:
                self._min_tau = self._max_tau / (2 * self._dim)
            self._heuristic_table[np.where(self._heuristic_table < self._min_tau)] = self._min_tau


    def _gradient(self, solution, cost):
        """Calculate the gradient for a single example."""
        sol_len = len(solution)
        # add 1/(path len) to each edge
        grad = dok_matrix(self._heuristic_table.shape)
        for k in range(sol_len):
            n1, n2 = solution[(k)%sol_len], solution[(k+1)%sol_len]
            grad[n1, n2] += 1 / cost
            # grad[n2, n1] += 1 / cost
        return grad

    def _buffer_gradient(self, index):
        """Calculate the gradient for a single example."""
        cost = self._replay_buffer_fit[index]
        # add 1/(path len) to each edge
        solution = self._replay_buffer[index]
        return self._gradient(solution, cost)

    def _gradient_update(self) -> None:
        """Take an gradient step."""
        if self._replay_rule != "none":
            buffer_sample = np.array(self._rng.integers(0, self._replay_size, self._update_size)).flatten() # sample with replacement
        else:
            buffer_sample = np.arange(self._replay_size)
        tot_grad = np.zeros([self._dim, self._dim])
        for i in buffer_sample:
            tot_grad += self._buffer_gradient(i)
        tot_grad = tot_grad/self._update_size

        self._heuristic_table = (1 - self._evap_rate) * self._heuristic_table + self._evap_rate * tot_grad
        self._minmax()


    def _get_candiates(self, taboo_set: set[int] | list[int]) -> list:
        """Get the availible nodes that are not in the taboo list."""
        return list(set(range(self._dim)) - set(taboo_set))

    def _single_solution(self):
        """Find a path for a single path."""
        solution = [self._rng.integers(self._dim)]
        for k in range(self._dim - 1):
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = np.array(self._prob_matrix[solution[k], allow_list]).flatten()
            prob = prob / prob.sum()
            next_point = self._rng.choice(allow_list, p=prob) # roulette selection
            solution.append(next_point)
        cost = self.func(self.distance_matrix, solution)
        grad = self._gradient(solution, cost)
        return np.array(solution, dtype=int), grad, cost

    def _prob_rule_update(self) -> None:
        """Update the probability matrix."""
        self._prob_matrix = np.multiply(np.power(self._heuristic_table, self._alpha),
                                        np.power(self._prob_bias, self._beta))

    def _prob_rule_update_softmax(self) -> None:
        """Update the probability matrix."""
        self._prob_matrix = np.exp(np.multiply(np.power(self._heuristic_table, self._alpha),
                                               np.power(self._prob_bias, self._beta)))

    def _prob_rule_node(self, node: int, taboo_list: set[int] | list[int]) -> float:
        """Return the probability of an action given a state and history."""
        probs = np.zeros(self._dim)
        allow_list = self._get_candiates(taboo_set=taboo_list)
        probs[allow_list] = self._prob_matrix[node][allow_list]
        probs /= probs.sum()
        return probs[node]

    def _elite_replay_rule(self, new_fitness, new_solutions, new_grads,  new_dist) -> None:
        """Keep the most recent `self._replay_size` elements in the `self._replay_buffer` and `self._replay_buffer_fit`."""
        full_fitness = np.concatenate([new_fitness, self._replay_buffer_fit])
        keep_indices = np.argpartition(full_fitness, kth=self._replay_size)[:self._replay_size]
        self._replay_buffer_fit = full_fitness[keep_indices]

        if new_solutions is not None:
            full_buffer = np.concatenate([new_solutions, self._replay_buffer])
            self._replay_buffer = full_buffer[keep_indices]

        if new_grads is not None:
            full_grads = np.concatenate([new_grads, self._replay_buffer_grads])
            self._replay_buffer_grads = full_grads[keep_indices]

        if new_dist is not None:
            full_probs = np.concatenate([new_dist, self._replay_buffer_probs])
            self._replay_buffer_probs = full_probs[keep_indices]

    def _evict_replay_rule(self, new_fitness, new_solutions, new_grads, new_dist) -> None:
        """Keep the best `self._replay_size` elements in the `self._replay_buffer` and `self._replay_buffer_fit`."""
        full_fitness = np.concatenate([new_fitness, self._replay_buffer_fit])
        keep_indices = np.arange(0, self._replay_size)
        self._replay_buffer_fit = full_fitness[keep_indices]

        if new_solutions is not None:
            full_buffer = np.concatenate([new_solutions, self._replay_buffer])
            self._replay_buffer = full_buffer[keep_indices]

        if new_grads is not None:
            full_grads = np.concatenate([new_grads, self._replay_buffer_grads])
            self._replay_buffer_grads = full_grads[keep_indices]

        if new_dist is not None:
            full_probs = np.concatenate([new_dist, self._replay_buffer_probs])
            self._replay_buffer_probs = full_probs[keep_indices]

    def _none_replay_rule(self, new_fitness, new_solutions, new_grads, new_dist) -> None:
        """Keep the best `self._replay_size` elements in the `self._replay_buffer` and `self._replay_buffer_fit`."""
        self._replay_buffer_fit = new_fitness
        if new_solutions is not None:
            self._replay_buffer = new_solutions

        if new_grads is not None:
            self._replay_buffer_grads = new_grads

        if new_dist is not None:
            self._replay_buffer_probs = new_dist


    def _add_replay_buffer(self, new_fitness,  new_solutions, new_grads, new_dist = None) -> None:
        """Add the provided list to the replay buffer, prefering new values when relevant."""
        match self._replay_rule:
            case "none":
                self._replay_buffer = new_solutions
                self._replay_buffer_fit = new_fitness
                self._replay_buffer_grads = new_grads
                if new_dist is not None:
                    self._replay_buffer_probs = new_dist
            case "elite":
                self._elite_replay_rule(new_fitness, new_solutions, new_grads, new_dist)
            case "evict":
                self._evict_replay_rule(new_fitness, new_solutions, new_grads, new_dist)

    def get_solution(self, start: int | None = None, seed: int | None =  None) -> tuple[float, np.ndarray]:
        if start is None:
            rng = np.random.default_rng(seed = seed or self._seed)
            start = rng.integers(self._dim)
        assert start >= 0
        path = [start]
        for _ in range(self._dim - 1):
            allow_list = self._get_candiates(path)
            next_point = allow_list[self._prob_matrix[path[-1], allow_list].argmax()]
            path.append(next_point)
        return self.func(self.distance_matrix, path), np.array(path)

    def _get_best(self, n: int = 1, sort: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return the n best values in the replay buffer."""
        # assert isinstance(sort, bool)

        if n == 1:
            index = np.array(self._replay_buffer_fit).argmin()
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
            gen_solutions = []
            gen_fitness = []
            gen_grads = []
            for _ in range(self._size_pop):
                sol, grad, cost = self._single_solution()
                gen_solutions.append(sol)
                gen_fitness.append(cost)
                gen_grads.append(grad)
            self._add_replay_buffer(gen_fitness, gen_solutions, gen_grads)

            self._gradient_update()

            # Get best solution and save it
            y, x = self._get_best(n=1)
            if y <= self.current_best_Y:
                self.current_best_Y, self.current_best_X = float(y), x
            if not self.slim:
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
              # beta          = 0,
              size_pop      = 2,
              replay_rule   = "global_best",
              slim = False,
              seed          = seed,
              )
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, "MINMAX " + aco._name_

def run_model2(distance_matrix, seed):
    aco = ACO(distance_matrix,
              # evap_rate     = 0.1,
              # alpha         = 1,
              # beta          = 2,
              size_pop      = 2,
              replay_rule   = "global_best",
              slim = False,
              seed          = seed,
              softmax       = True)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, "Softmax MINMAX " + aco._name_



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pgaco.utils import get_graph, plot, parallel_runs
    size = 200
    runs = 5
    max_iter = 150
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
