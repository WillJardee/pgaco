"""
Policy Gradient Ant Colony Optimization (PGACO) implementation for solving TSP problems.

This module provides classes and functions to implement the PGACO algorithm
that combines Policy Gradient methods with Ant Colony Optimization,
particularly focused on solving the Traveling Salesman Problem (TSP).

Classes:
    PGACO_LOG: Policy Gradient ACO with log-gradient.

The ACO algorithm simulates the behavior of ants to find optimal paths in a
graph, which can be applied to various shortest problems.

Example:
    pgaco_log = PGACO_LOG(problem_instance, size_pop = 100, learning_rate=10_000)
    pgaco_log.run(max_iter=1000)
"""


import numpy as np

from pgaco.model.ACO import ACO_TSP


class PGACO_LOG(ACO_TSP):
    """Implementation of ACO with log policy gradient update.

    Attributes
    ----------
        See parent's documentation
        learning_rate = learning rate for the gradient update.
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 **kwargs) -> None:
        """Class specific params."""
        self.allowed_params = {"learning_rate", "value_param",
                               "advantage_func", "annealing_factor"}
        super().__init__(distance_matrix, **self._passkwargs(**kwargs))
        self._name_ = "Log Policy"
        self._learning_rate = kwargs.get("learning_rate", 100)
        # self._running_gradient = np.zeros((self._dim, self._dim))
        # self._replay_buffer_grads = np.array([self._running_gradient for _ in range(self._replay_size)])
        self._adv_func = kwargs.get("advantage_func", "local")
        self._annealing_factor = kwargs.get("annealing_factor", 0.01)

    def _gradient(self, solution, cost) -> np.ndarray:
        """Take the sum of all gradients in the replay buffer."""
        # add 1/(path len) to each edge
        grad = np.ones(self._heuristic_table.shape)
        for k in range(len(solution) - 1):
            n1, n2 = solution[k], solution[k+1]
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = self._prob_matrix[solution[k], allow_list]
            prob = prob / prob.sum()
            advantage = self._advantage(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)
            grad[n1, n2] += self._alpha * advantage / self._heuristic_table[n1, n2]
            for point, prob_val in zip(allow_list, prob):
                grad[n1, point] -= self._alpha * advantage /self._heuristic_table[n1, point] * prob_val
        return grad

    def _advantage_local(self, current_point, next_point, allow_list):
        """Advantage function of the form: 1/C(x) - Avg(1/C(x))."""
        return 1/self.distance_matrix[current_point, next_point] - np.average(1/self.distance_matrix[current_point, allow_list])

    def _advantage_path(self, **kwargs):
        """
        Calculate the advantage function.

        This function computes the advantage using the formula:
        1/C(s_{t}) - Avg(1/C(s_{t-1}))

        Returns
        -------
        float
            The calculated advantage value.

        """
        pass

    def _advantage(self, **kwargs):
        """Return advantage function defined in `advantage`."""
        valid_adv = {"local", "path"}
        match self._adv_func:
            case "local":
                return self._advantage_local(**kwargs)
            case "path":
                return self._advantage_path(**kwargs)
            case _:
                raise ValueError(f"Advantage function not defined. Vaild choices: {valid_adv}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    size = 50
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))

    print("Running GPACO with log update")
    ACA_runs = []

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = PGACO_LOG(distance_matrix,
                        max_iter = iterations,
                        save_file = save_file)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
