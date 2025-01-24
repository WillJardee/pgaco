"""
Policy Gradient Ant Colony Optimization (PGACO) implementation for solving TSP problems.

This module provides classes and functions to implement the PGACO algorithm
that combines Policy Gradient methods with Ant Colony Optimization,
particularly focused on solving the Traveling Salesman Problem (TSP).

Classes:
    PGACO_RATIO: Policy Gradient ACO with policy ratio.

The ACO algorithm simulates the behavior of ants to find optimal paths in a
graph, which can be applied to various shortest problems.

Example:
    pgaco_ratio = PGACO_RATIO(problem_instance, epsilon=-1, size_pop = 100, learning_rate=10_000)
    pgaco_ratio.run(max_iter=1000)
"""


import numpy as np
from typing import Callable, Iterable

from pgaco.models import ACOSGD, path_len


class ACOPG(ACOSGD):
    """Implementation of ACA with prob ratio policy gradient update; clipping is on by default."""

    def __init__(self,
                 distance_matrix,
                 func: Callable[[np.ndarray, Iterable], float] = path_len,
                 *,
                 evap_rate: float = 3,
                 epsilon: float = 0.2,
                 **kwargs,
                 ) -> None:
        """Class specific params."""
        super().__init__(distance_matrix, func, **kwargs)
        self._prob_table_last_gen = self._prob_matrix
        self._epsilon = epsilon
        self._clip = False if self._epsilon == -1 else True
        self._name_ = "Policy Ratio"

    @property
    def _epsilon(self):
        return self.__epsilon

    @_epsilon.setter
    def _epsilon(self, epsilon):
        assert self._between(epsilon, lower=0, upper=1) or epsilon == -1
        self.__epsilon = float(epsilon)

    def _single_solution(self):
        """Find a path for a single path."""
        solution = [self._rng.integers(self._dim)]
        dist = []
        # grad = dok_matrix((self._dim, self._dim), dtype=np.float64)
        sub_term = 0
        for k in range(self._dim - 1):
            # Forward
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = np.array(self._prob_matrix[solution[k], allow_list]).flatten()
            prob = prob / prob.sum()
            index = self._rng.choice(len(allow_list), p=prob) # roulette selection
            solution.append(allow_list[index])
            dist.append(prob[index])

        cost = self.func(self.distance_matrix, solution)
        return np.array(solution, dtype=int), None, cost, dist


    def _gradient(self, solution, prev_dist):
        running_grad = np.zeros([self._dim, self._dim])
        sub_term = np.zeros([self._dim, self._dim])
        for k in range(len(solution) - 1):
            n1, n2 = solution[k], solution[k+1]
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            n2_index = allow_list.index(n2)
            prob = np.array(self._prob_matrix[n1, allow_list]).flatten()
            prob_normed = prob / prob.sum()
            importance_ratio = prob_normed[n2_index]/prev_dist[k]

            advantage = self._advantage(k, solution, allow_list)

            coef = importance_ratio * advantage * self._alpha
            prob_normed[n2_index] -= 1
            ##! Inner needs to be broadcast to the shape of the map; consider a sparse matrix
            inner = prob_normed * 1/prob
            if self._clip:
                if advantage > 0:
                    if importance_ratio > (1 + self._epsilon):
                        continue
                elif advantage < 0:
                    if importance_ratio < (1 - self._epsilon):
                        continue
            for index, n2 in zip(range(len(allow_list)), allow_list):
                running_grad[n1, n2] += coef * inner[index]
        return -1 * running_grad

    def _buffer_gradient(self, index):
        solution = self._replay_buffer[index]
        prev_dist = self._replay_buffer_probs[index]
        return self._gradient(solution, prev_dist)


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

    # This is here temporarily; I did not want to spend the time to refactor
    # the original aco to accept the distributions
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
            gen_dists = []
            for _ in range(self._size_pop):
                sol, grad, cost, dist = self._single_solution()
                gen_solutions.append(sol)
                gen_fitness.append(cost)
                gen_grads.append(grad)
                gen_dists.append(dist)
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

            self._minmax()

            # Save check
            if self._iteration % self._checkpoint_res == 0:
                if self._savefile is not None:
                    self._save()
                if self._checkpointfile is not None:
                    self._checkpoint()

        return float(self.current_best_Y), self.current_best_X




def run_model1(distance_matrix, seed):
    aco = ACOPG(distance_matrix,
                size_pop      = 2,
                slim = False,
                evap_rate = 0.001,
                seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_

def run_model2(distance_matrix, seed):
    aco = ACOPG(distance_matrix,
                size_pop      = 2,
                epsilon       = -1,
                slim = False,
                seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, "Clipped " + aco._name_

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pgaco.utils import get_graph, plot, parallel_runs
    size = 20
    runs = 1
    max_iter = 150
    distance_matrix = get_graph(size)

    print("running ACOPG")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model1, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="cyan", label=aco_name)
    plot(aco_policy_runs, color="blue", label=aco_name + " policy")

    print("running ACOPPO")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model2, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="green", label=aco_name)
    plot(aco_policy_runs, color="lime", label=aco_name + " policy")

    plt.legend()
    plt.tight_layout()
    plt.show()

    pass
