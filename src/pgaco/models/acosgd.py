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
from typing import Callable, Iterable
from enum import Enum
import numpy as np
from pgaco.models import ACO, path_len


class ACOSGD(ACO):
    """Implementation of ACO with log policy gradient update.

    Attributes
    ----------
        See parent's documentation
        learning_rate = learning rate for the gradient update.
    """

    class Advantage(Enum):
        ADVANTAGE_LOCAL         = "local"
        ADVANTAGE_PATH          = "path"
        QUALITY                 = "quality"
        REWARD                  = "reward"
        REWARD_TO_GO            = "reward-to-go"
        REWARD_TO_GO_BASELINE   = "reward-to-go-baseline"
        REWARD_BASELINE         = "reward-baseline"
        REWARD_DECAY            = "reward-decay"
        REWARD_ENTROPY          = "reward-entropy"

    def __init__(self,
                 distance_matrix: np.ndarray,
                 func: Callable[[np.ndarray, Iterable], float] = path_len,
                 *,
                 advantage_func: str = "reward-baseline",
                 regularizer: str | None = "l2",
                 annealing_factor: float = 0.01,
                 exact_grad: bool = False,
                 **kwargs) -> None:
        """Class specific params."""
        super().__init__(distance_matrix, func, **kwargs)
        self._adv_func = advantage_func
        self._regularizer = regularizer
        self._annealing_factor = annealing_factor
        self._exact_grad = exact_grad
        self._replay_buffer_grads = np.array([{"sub_term" : 0} for _ in range(self._replay_size)])
        self._depth = 10
        self._avg_cost = [np.mean([self.func(self.distance_matrix, s[:k]) for s in self._replay_buffer]) for k in range(self._dim)]
        self._name_ = "ACO-SGD"

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
        assert adv_func in [item.value for item in self.Advantage]
        self.__adv_func = self.Advantage(adv_func)

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
        grad = {}
        sub_term = 0
        for k in range(self._dim - 1):
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = np.array(self._prob_matrix[solution[k], allow_list]).flatten()
            prob = prob / prob.sum()
            next_point = self._rng.choice(allow_list, p=prob) # roulette selection
            solution.append(next_point)
            n1, n2 = solution[k], solution[k+1]

            # advantage = self._advantage(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)

            advantage = self._advantage(k, solution, allow_list)

            grad[(k, k+1)] = self._alpha * advantage / self._heuristic_table[k, k+1]
            if self._exact_grad:
                for point, prob_val in zip(allow_list, prob):
                    # TODO: Unsure if the prob_val should be there
                    # grad[k, k+1] -= self._alpha * advantage /self._heuristic_table[k, point] * prob_val
                    grad[(k, k+1)] = - self._alpha * advantage /self._heuristic_table[k, point] * prob_val
            else:
                sub_term += self._alpha * advantage /self._heuristic_table[k].max()
        # if self._exact_grad:
        #     grad = np.matrix(grad.todense())
        grad["sub_term"] = sub_term
        cost = self.func(self.distance_matrix, solution)
        return np.array(solution, dtype=int), grad, cost

    def _gradient(self, solution):
        running_grad = np.zeros([self._dim, self._dim])
        sub_term = np.zeros([self._dim, self._dim])
        for k in range(len(solution) - 1):
            n1, n2 = solution[k], solution[k+1]
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            n2_index = allow_list.index(n2)
            prob = np.array(self._prob_matrix[n1, allow_list]).flatten()
            prob_normed = prob / prob.sum()

            advantage = self._advantage(k, solution, allow_list)

            coef = advantage * self._alpha
            prob_normed[n2_index] -= 1
            ##! Inner needs to be broadcast to the shape of the map; consider a sparse matrix
            inner = prob_normed * 1/prob
            # if self._clip:
            #     if advantage > 0:
            #         if importance_ratio > (1 + self._epsilon):
            #             continue
            #     elif advantage < 0:
            #         if importance_ratio < (1 - self._epsilon):
            #             continue
            for index, n2 in zip(range(len(allow_list)), allow_list):
                running_grad[n1, n2] += coef * inner[index]
        return -1 * running_grad

    def _buffer_gradient(self, index):
        solution = self._replay_buffer[index]
        return self._gradient(solution)

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


    # def _gradient_update(self) -> None:
    #     """Take an gradient step."""
    #     tot_grad = np.zeros(self._heuristic_table.shape)
    #     # for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
    #     #     tot_grad += self._gradient(solution, cost)
    #     # tot_grad = tot_grad/self._replay_size
    #     for grad in self._replay_buffer_grads:
    #         for coord, val in zip(grad.keys(), grad.values()):
    #             if coord == "sub_term":
    #                 continue
    #             tot_grad[coord[0], coord[1]] = val
    #         tot_grad -= grad["sub_term"]
    #     tot_grad = tot_grad/self._replay_size
    #
    #
    #     if self._regularizer == "l2":
    #         self._heuristic_table = self._evap_rate * self._heuristic_table - (1-self._evap_rate) * tot_grad
    #     else:
    #         self._heuristic_table = self._heuristic_table - (1-self._evap_rate) * tot_grad
    #     self._minmax()

    def _advantage_local(self, current_point, next_point, allow_list):
        """Advantage function of the form: 1/C(x) - Avg(1/C(x))."""
        return self._heuristic_table[current_point, next_point] - np.average(self._heuristic_table[current_point, allow_list])

    def _advantage_path(self, path):
        """Advantage function of the form: 1/C(s_t) - Avg(1/C(s_t))."""
        return 1/self.func(self.distance_matrix, path) - 1/self._avg_cost[len(path)-1]

    def _quality(self, current_point, next_point):
        """Quality Function (_heuristic_table)."""
        return self._heuristic_table[current_point, next_point]

    def _reward(self, current_point, next_point):
        """Reward function (1/C(x))"""
        return 1/self.distance_matrix[current_point, next_point]

    def _reward_baselined(self, current_point, next_point, allow_list):
        """Advantage function of the form: 1/C(x) - Avg(1/C(x))."""
        baseline = 1/self.distance_matrix[current_point, next_point] - 1/np.average(self.distance_matrix[current_point, allow_list])
        return baseline

    def _reward_to_go(self, current_point, trace, depth):
        """return the sum of the rewards accumulated from current_point, following trace, for length depth."""
        depth = depth if depth < len(trace) else len(trace)
        reward_to_go = sum([1/self.distance_matrix[trace[i], trace[i+1]] for i in range(depth-1)])
        return reward_to_go/depth

    def _reward_to_go_baselined(self, current_point, trace, allow_list, depth):
        """return the sum of the rewards accumulated from current_point, following trace, for length depth, minues the average cost of edges accessible."""
        depth = depth if depth < len(trace) else len(trace)
        reward_to_go = 1/(sum([self.distance_matrix[trace[i], trace[i+1]] for i in range(depth-1)])/depth)
        baseline = 1/np.average(self.distance_matrix[current_point, allow_list])
        return reward_to_go - baseline

    def _reward_decay(self, current_point, trace, decay, depth):
        """return the sum of the sum of the rewards going back up to depth steps, each multiplied by value of decay. This is the bellman reward structure."""
        raise NotImplementedError()

    def _reward_entropy(self, base_reward, entropy_weight):
        """return the base_reward calculation plus a regularized term from entropy_weight"""
        raise NotImplementedError()

    def _advantage(self, k, solution, allow_list):
        """Return advantage function defined in `advantage`."""
        n1 = solution[k]
        n2 = solution[k+1]
        if self._adv_func == None:
            advantage = 1
        elif self._adv_func == self.Advantage.ADVANTAGE_LOCAL:
            advantage = self._advantage_local(n1, n2, allow_list)
        elif self._adv_func == self.Advantage.ADVANTAGE_PATH:
            advantage = self._advantage_path(solution)
        elif self._adv_func == self.Advantage.QUALITY:
            advantage = self._quality(n1, n2)
        elif self._adv_func == self.Advantage.REWARD:
            advantage = self._reward(n1, n2)
        elif self._adv_func == self.Advantage.REWARD_TO_GO:
            advantage = self._reward_to_go(n1, solution[k:], self._depth)
        elif self._adv_func == self.Advantage.REWARD_TO_GO_BASELINE:
            advantage = self._reward_to_go_baselined(n1, solution[k:], allow_list, self._depth)
        elif self._adv_func == self.Advantage.REWARD_BASELINE:
            advantage = self._reward_baselined(n1, n2, allow_list)
        elif self._adv_func == self.Advantage.REWARD_DECAY:
            advantage = self._reward_decay(n1, solution[k:], None, self._depth)
        elif self._adv_func == self.Advantage.REWARD_ENTROPY:
            advantage = self._reward_entropy(None, None)
        else:
            raise NotImplementedError(f"Advantage function {self._adv_func} is not valid")
        return advantage

    # def take_step(self, steps=1) -> tuple[float, np.ndarray]:
    #     if self._adv_func in ["path"]:
    #         self._avg_cost = [np.mean([self.func(s[:k]) for s in self._replay_buffer]) for k in range(self._dim)]
    #     score, solution = super().take_step(steps)
    #     return score, solution
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
                sol, grad, cost, = self._single_solution()
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

            self._minmax()

            # Save check
            if self._iteration % self._checkpoint_res == 0:
                if self._savefile is not None:
                    self._save()
                if self._checkpointfile is not None:
                    self._checkpoint()

        return float(self.current_best_Y), self.current_best_X



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
    plot(aco_policy_runs, color="blue", label=aco_name + " policy")

    print("running ACOSGD")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model2, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="green", label=aco_name)
    plot(aco_policy_runs, color="lime", label=aco_name + " policy")

    plt.legend()
    plt.tight_layout()
    plt.show()

    pass
