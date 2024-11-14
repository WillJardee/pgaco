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
from typing import Callable, Iterable
from pgaco.models import ACO, path_len


class ACOSGD(ACO):
    """Implementation of ACO with log policy gradient update.

    Attributes
    ----------
        See parent's documentation
        learning_rate = learning rate for the gradient update.
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 func: Callable[[np.ndarray, Iterable], float] = path_len,
                 *,
                 advantage_func: str = "quality",
                 regularizer: str | None = "l2",
                 annealing_factor: float = 0.01,
                 exact_grad: bool = False,
                 **kwargs) -> None:
        """Class specific params."""
        self._name_ = "ACO-SGD"
        super().__init__(distance_matrix, func, **kwargs)
        self._adv_func = advantage_func
        self._regularizer = regularizer
        self._annealing_factor = annealing_factor
        self._exact_grad = exact_grad
        self._replay_buffer_grads = np.array([{"sub_term" : 0} for _ in range(self._replay_size)])

    """The following is collection of parameter validation rules."""
    @property
    def _evap_rate(self):
        return self.__evap_rate

    @_evap_rate.setter
    def _evap_rate(self, evap_rate):
        assert self._between(evap_rate, lower=0, inclusive=True)
        self.__evap_rate = float(evap_rate)

    @property
    def _adv_func(self):
        return self.__adv_func

    @_adv_func.setter
    def _adv_func(self, adv_func):
        assert adv_func in ["quality", "local", "reward", "path", None]
        self.__adv_func = adv_func

    @property
    def _regularizer(self):
        return self.__regularizer

    @_regularizer.setter
    def _regularizer(self, regularizer):
        assert regularizer in ["l2", None]
        self.__regularizer = regularizer

    @property
    def _annealing_factor(self):
        return self.__annealing_factor

    @_annealing_factor.setter
    def _annealing_factor(self, annealing_factor):
        assert self._between(annealing_factor, lower=0, upper=1)
        self.__annealing_factor = float(annealing_factor)

    def _single_solution(self):
        """Find a path for a single path."""
        solution = [self._rng.integers(self._dim)]
        # grad = dok_matrix((self._dim, self._dim), dtype=np.float64)
        grad = 0
        for k in range(self._dim - 1):
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = np.array(self._prob_matrix[solution[k], allow_list]).flatten()
            prob = prob / prob.sum()
            next_point = self._rng.choice(allow_list, p=prob) # roulette selection
            solution.append(next_point)
        #
        #     advantage = self._advantage(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)
        #     grad[(k, k+1)] = self._alpha * advantage / self._heuristic_table[k, k+1]
        #     if self._exact_grad:
        #         for point, prob_val in zip(allow_list, prob):
        #             # TODO: Unsure if the prob_val should be there
        #             # grad[k, k+1] -= self._alpha * advantage /self._heuristic_table[k, point] * prob_val
        #             grad[(k, k+1)] = - self._alpha * advantage /self._heuristic_table[k, point] * prob_val
        #     else:
        #         sub_term += self._alpha * advantage /self._heuristic_table[k].max()
        # # if self._exact_grad:
        # #     grad = np.matrix(grad.todense())
        # grad["sub_term"] = sub_term
        cost = self.func(self.distance_matrix, solution)
        return np.array(solution, dtype=int), grad, cost

    def _gradient(self, solution) -> np.ndarray:
        """Take the sum of all gradients in the replay buffer."""
        # add 1/(path len) to each edge
        grad = np.zeros(self._heuristic_table.shape)
        for k in range(len(solution) - 1):
            n1, n2 = solution[k], solution[k+1]
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = np.array(self._prob_matrix[solution[k], allow_list]).flatten()
            prob = prob / prob.sum()
            advantage = self._advantage(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)
            grad[n1, n2] += self._alpha * advantage / self._heuristic_table[n1, n2]
            for point, prob_val in zip(allow_list, prob):
                grad[n1, point] -= self._alpha * advantage /self._heuristic_table[n1, point] * prob_val
        return grad

    def _gradient_update(self) -> None:
        """Take an gradient step."""
        tot_grad = np.zeros(self._heuristic_table.shape)
        for solution, _ in zip(self._replay_buffer, self._replay_buffer_fit):
            tot_grad += self._gradient(solution)
        tot_grad = tot_grad/self._replay_size
        # for grad in self._replay_buffer_grads:
        #     for coord, val in zip(grad.keys(), grad.values()):
        #         if coord == "sub_term":
        #             continue
        #         tot_grad[coord[0], coord[1]] = val
        #     tot_grad -= grad["sub_term"]
        tot_grad = tot_grad/self._replay_size


        if self._regularizer == "l2":
            self._heuristic_table = self._evap_rate * self._heuristic_table - (1-self._evap_rate) * tot_grad
        else:
            self._heuristic_table = self._heuristic_table - (1-self._evap_rate) * tot_grad
        self._minmax()

    def _advantage_local(self, current_point, next_point, allow_list):
        """Advantage function of the form: 1/C(x) - Avg(1/C(x))."""
        return 1/self.distance_matrix[current_point, next_point] - np.average(1/self.distance_matrix[current_point, allow_list])

    def _advantage_path(self, path):
        """Advantage function of the form: 1/C(s_t) - Avg(1/C(s_t))."""
        return 1/self.func(path) - 1/self._avg_cost[len(path)-1]

    def _quality(self, current_point, next_point):
        """Quality Function (_heuristic_table)."""
        return self._heuristic_table[current_point, next_point]

    def _reward(self, current_point, next_point):
        """Reward function (1/C(x))"""
        return 1/self.distance_matrix[current_point, next_point]

    def _advantage(self, **kwargs):
        """Return advantage function defined in `advantage`."""
        if self._adv_func is None:
            return 1
        match self._adv_func:
            case "local":
                return self._advantage_local(kwargs.get("current_point"), kwargs.get("next_point"), kwargs.get("allow_list"))
            case "path":
                return self._advantage_path(kwargs.get("path"))
            case "quality":
                return self._quality(kwargs.get("current_point"), kwargs.get("next_point"))
            case "reward":
                return self._reward(kwargs.get("current_point"), kwargs.get("next_point"))
            case _:
                raise ValueError(f"Advantage function not defined. Vaild choices: {valid_adv}")

    def take_step(self, steps=1) -> tuple[float, np.ndarray]:
        if self._adv_func in ["path"]:
            self._avg_cost = [np.mean([self.func(s[:k]) for s in self._replay_buffer]) for k in range(self._dim)]
        score, solution = super().take_step(steps)
        return score, solution


def run_model1(distance_matrix, seed):
    aco = ACOSGD(distance_matrix,
                 size_pop      = 2,
                 slim = False,
                 regularizer   = None,
                 exact_grad=True,
                 seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_ + " w/ exact"

def run_model2(distance_matrix, seed):
    aco = ACOSGD(distance_matrix,
                 size_pop      = 2,
                 regularizer   = None,
                 slim = False,
                 # exact_grad=True,
                 seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pgaco.utils import get_graph, plot, parallel_runs
    size = 20
    runs = 5
    max_iter = 150
    distance_matrix = get_graph(size)

    print("running ACOSGD w/ regularizer")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model1, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="cyan", label=aco_name)
    # plot(aco_policy_runs, color="blue", label=aco_name + " policy")

    print("running ACOSGD")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model2, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="green", label=aco_name)
    # plot(aco_policy_runs, color="lime", label=aco_name + " policy")

    plt.legend()
    plt.tight_layout()
    plt.show()

    pass
