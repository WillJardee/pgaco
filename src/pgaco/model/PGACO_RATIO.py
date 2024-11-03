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

from pgaco.model.PGACO_LOG import PGACO_LOG


class PGACO_RATIO(PGACO_LOG):
    """Implementation of ACA with prob ratio policy gradient update; clipping is on by default."""

    def __init__(self, distance_matrix, **kwargs) -> None:
        """Class specific params."""
        self.allowed_params = {"epsilon"}
        super().__init__(distance_matrix, **self._passkwargs(**kwargs))
        self._name_ = "Policy Ratio"
        self._prob_table_last_gen = self._prob_matrix
        self._epsilon = kwargs.get("epsilon", 0.1)
        self._clip = False if self._epsilon == -1 else True

    def _gradient_add(self, current_point, next_point, allow_list, prob, advantage) -> None:
        """Add a value to the running gradient."""
        norm_term = self._prob_matrix[current_point, allow_list].sum()
        norm_term_old = self._prob_table_last_gen[current_point, allow_list].sum()
        prob_chosen = self._prob_matrix[current_point, next_point]/norm_term
        prob_chosen_old = self._prob_table_last_gen[current_point, next_point]/norm_term_old
        prob_ratio = (prob_chosen/prob_chosen_old)

        if self._clip:
            if advantage > 0:
                if prob_ratio > (1 + self._epsilon):
                    return
            elif advantage < 0:
                if prob_ratio < (1 - self._epsilon):
                    return

        self._running_gradient[current_point, next_point] += self._alpha * advantage * prob_ratio
        for point, prob_val in zip(allow_list, prob):
            self._running_gradient[current_point, point] -= self._alpha * advantage * prob_ratio * prob_val

    def _gradient_update(self) -> None:
        super()._gradient_update()
        self._prob_table_last_gen = self._prob_matrix


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    size = 50
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))

    print("Running GPACO with ratio update")
    ACA_runs = []

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = PGACO_RATIO(distance_matrix,
                          max_iter = iterations,
                          save_file = save_file)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"PGACO_ratio: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
