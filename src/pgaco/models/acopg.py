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
                 epsilon: float = 0.1,
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
        # grad = dok_matrix((self._dim, self._dim), dtype=np.float64)
        grad = {}
        sub_term = 0
        for k in range(self._dim - 1):
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = np.array(self._prob_matrix[solution[k], allow_list]).flatten()
            prob = prob / prob.sum()
            next_point = self._rng.choice(allow_list, p=prob) # roulette selection
            solution.append(next_point)
            n1, n2 = solution[-1], solution[-2]

            advantage = self._advantage_local(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)
            norm_term = self._prob_matrix[n1, allow_list].sum()
            norm_term_old = self._prob_table_last_gen[n1, allow_list].sum()
            prob_chosen = self._prob_matrix[n1, n2]/norm_term
            prob_chosen_old = self._prob_table_last_gen[n1, n2]/norm_term_old
            prob_ratio = (prob_chosen/prob_chosen_old)

            advantage = self._advantage(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)
            if self._clip:
                if advantage > 0:
                    if prob_ratio > (1 + self._epsilon):
                        continue
                elif advantage < 0:
                    if prob_ratio < (1 - self._epsilon):
                        continue

            grad[(k, k+1)] = self._alpha * advantage * prob_ratio
            if self._exact_grad:
                for point, prob_val in zip(allow_list, prob):
                    # TODO: Unsure if the prob_val should be there
                    grad[(k, k+1)] = - self._alpha * advantage * prob_ratio * prob_val
            else:
                sub_term += self._alpha * advantage * prob_ratio * prob.max()
        # if self._exact_grad:
        #     grad = np.matrix(grad.todense())
        grad["sub_term"] = sub_term
        cost = self.func(self.distance_matrix, solution)
        return np.array(solution, dtype=int), grad, cost

    # def _gradient(self, solution, cost) -> np.ndarray:
    #     """Take the sum of all gradients in the replay buffer."""
    #     # add 1/(path len) to each edge
    #     grad = np.zeros(self._heuristic_table.shape)
    #     for k in range(len(solution) - 1):
    #         n1, n2 = solution[k], solution[k+1]
    #         allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
    #         prob = self._prob_matrix[solution[k], allow_list]
    #         prob = prob / prob.sum()
    #         advantage = self._advantage_local(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)
    #
    #         norm_term = self._prob_matrix[n1, allow_list].sum()
    #         norm_term_old = self._prob_table_last_gen[n1, allow_list].sum()
    #         prob_chosen = self._prob_matrix[n1, n2]/norm_term
    #         prob_chosen_old = self._prob_table_last_gen[n1, n2]/norm_term_old
    #         prob_ratio = (prob_chosen/prob_chosen_old)
    #
    #         if self._clip:
    #             if advantage > 0:
    #                 if prob_ratio > (1 + self._epsilon):
    #                     continue
    #             elif advantage < 0:
    #                 if prob_ratio < (1 - self._epsilon):
    #                     continue
    #
    #         grad[n1, n2] += self._alpha * advantage * prob_ratio
    #         for point, prob_val in zip(allow_list, prob):
    #             grad[n1, point] -= self._alpha * advantage * prob_ratio * prob_val
    #     return grad

    def _gradient_update(self) -> None:
        super()._gradient_update()
        self._prob_table_last_gen = self._prob_matrix


def run_model1(distance_matrix, seed):
    aco = ACOPG(distance_matrix,
                size_pop      = 2,
                slim = False,
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
    runs = 5
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
