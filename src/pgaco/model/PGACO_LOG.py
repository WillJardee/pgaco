import numpy as np
from tqdm import tqdm
import networkx as nx
import ast
import pickle
try: from .ACO import ACO_TSP
except: from ACO import ACO_TSP

class PGACO_LOG(ACO_TSP):
    """Implementation of ACO with log policy gradient update

    Attributes:
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
        self._running_gradient = np.zeros((self._dim, self._dim))
        self._replay_buffer_grads = np.array([self._running_gradient for _ in range(self._replay_size)])
        self.value_matrix = np.zeros((self._dim-1))
        self.value_param = kwargs.get("value_param", 0.7)
        self._adv_func = kwargs.get("advantage_func", "local")
        self._annealing_factor = kwargs.get("annealing_factor", 0.01)

    def _passkwargs(self, **kwargs):
        passkwargs = kwargs.copy()
        for key in kwargs:
            if key in self.allowed_params:
                passkwargs.pop(key)
        return passkwargs

    def _gradient(self) -> np.ndarray:
        """Take the sum of all gradients in the replay buffer"""
        grad = np.zeros([self._dim, self._dim])
        for g in self._replay_buffer_grads:
            grad += g
        return grad / self._size_pop


    def _gradient_add(self, current_point, next_point, allow_list, prob, advantage) -> None:
        """Add a value to the running gradient."""
        self._running_gradient[current_point, next_point] += self._alpha * advantage / self._heuristic_table[current_point, next_point]
        for point, prob_val in zip(allow_list, prob):
            self._running_gradient[current_point, point] -= self._alpha * advantage /self._heuristic_table[current_point, point] * prob_val

    def _gradient_update(self) -> None:
        """Take an gradient step"""
        self._heuristic_table = self._heuristic_table + self._learning_rate * self._gradient()
        self._heuristic_table[np.where(self._heuristic_table < self._min_tau)] = self._min_tau

    def _add_replay_buffer(self, new_fitness, new_solutions, new_grads, sort: bool = True) -> None:
        """Add the provided list to the replay buffer, prefering new values when relevant"""
        if not self._replay: # if we are not doing replay, just return the values
            self._replay_buffer = new_solutions
            self._replay_buffer_fit = new_fitness
            return

        full_buffer = np.concatenate([new_solutions, self._replay_buffer])
        full_fitness = np.concatenate([new_fitness, self._replay_buffer_fit])
        full_grads = np.concatenate([new_grads, self._replay_buffer_grads])

        if sort:
            keep_indices = np.argsort(full_fitness)[:self._replay_size]
        else:
            keep_indices = np.argpartition(full_fitness, kth=self._replay_size)[:self._replay_size]
        self._replay_buffer = full_buffer[keep_indices]
        self._replay_buffer_fit = full_fitness[keep_indices]
        self._replay_buffer_grads = full_grads[keep_indices]


    def _advantage_local(self, **kwargs):
        """Advantage function of the form:
        1/C(x) - Avg(1/C(x))
        """
        current_point   = kwargs["current_point"]
        next_point      = kwargs["next_point"]
        allow_list      = kwargs["allow_list"]
        return 1/self.distance_matrix[current_point, next_point] - np.average(1/self.distance_matrix[current_point, allow_list])

    def _advantage_path(self, **kwargs):
        """Advantage function of the form
        1/C(s_{t}) - Avg(1/C(s_{t-1}))
        :returns: (float) calculated average

        """
        pass


    def _advantage(self, **kwargs):
        valid_adv = {"local", "path"}
        match self._adv_func:
            case "local":
                self._advantage_local(*kwargs)
            case "path":
                self._advantage_path(*kwargs)
            case _:
                raise ValueError(f"Advantage function not defined. Vaild choices: {valid_adv}")

    def _single_solution(self):
        """Find a path for a single path."""
        self._running_gradient = np.zeros([self._dim, self._dim])
        solution = [np.random.randint(self._dim)]
        for k in range(self._dim - 1):
            allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
            prob = self._prob_matrix[solution[k], allow_list]
            prob = prob / prob.sum()
            next_point = np.random.choice(allow_list, size=1, p=prob)[0] # roulette selection
            solution.append(next_point)

            advantage = self._advantage_local(current_point=solution[-2], next_point=solution[-1], allow_list=allow_list)
            self._gradient_add(solution[-2], solution[-1], allow_list, prob, advantage)
        return np.array(solution, dtype=int)

    def run(self, max_iter=None, metric= ["best"]):
        """Runs through solving the TSP."""
        if self.func is None or self.distance_matrix is None:
            raise ValueError(f"func and distance_matrix must be set to run {self._name_}")

        self.set_metrics(metric)
        self.metrics = metric
        for i in range(max_iter or self._max_iter):
            self.iteration = i
            self._prob_rule_update()

            # Generate solutions
            gen_solutions, gen_grads = [], []
            for _ in range(self._size_pop):
                gen_solutions.append(self._single_solution())
                gen_grads.append(self._running_gradient)
            gen_fitness = np.array([self.func(i) for i in gen_solutions])
            self._add_replay_buffer(gen_fitness, gen_solutions, gen_grads)

            self._gradient_update()

            # Get best solution and save it
            y, x = self._get_best(n=1)
            if y <= self.current_best_Y: self.current_best_Y, self.current_best_X = y, x
            self.generation_best_Y.append(y)

            # Save check
            if self.iteration % self._checkpoint_res == 0:
                if self.save_file is not None: self._save()
                if self.checkpoint_file is not None: self._checkpoint()
            self._learning_rate = self._learning_rate * (1 - self._annealing_factor)

        return self.current_best_Y, self.current_best_X


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    size = 50
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))

    print("Running GPACO with log update")
    ACA_runs = []

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = PolicyGradient3ACA(distance_matrix,
                      max_iter = iterations,
                      save_file = save_file)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
