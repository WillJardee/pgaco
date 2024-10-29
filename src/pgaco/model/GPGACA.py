import numpy as np
from tqdm import tqdm
import networkx as nx
import ast
import pickle
from .ACA import ACA_TSP

class PolicyGradient3ACA(ACA_TSP):
    """Implementation of ACA with log policy gradient update

    Attributes:
        distrtance_matrix = weighted adjacency matrix of the given TSP space.
        n_dim = number of nodes in the graph.
        func = function to optimize wrt.
        size_pop = number of ants.
        max_iter = maximum number of iterations.
        alpha = pheromone exponent.
        beta = bias exponent.
        rho = pheromone evaporation rate.
        min_tau = minimum pheromone value.
        prob_bias = probability bias rule.
        lamb = branching factor.
        save_file = path to save learned pheromone table.
        _checkpoint_file = path to save learned pheromone table.
        _checkpoint_res = checkpoint resolution.
        learning_rate = learning rate for the gradient update.
    """
    def __init__(self, func,
                 distance_matrix,
                 params: dict = {}) -> None:
        """Class specific params."""
        super().__init__(func, distance_matrix, params=params)
        self._name_ = "Log Policy"
        self.learning_rate = params.get("learning_rate", 1)
        self.Table_grad = np.zeros((self.n_dim, self.n_dim))
        self.value_matrix = np.zeros((self.n_dim-1))
        self.value_param = params.get("value_param", 0.7)

    def _delta_tau(self) -> np.ndarray:
        """Calculate the update rule."""
        return self.Table_grad


    def _phero_update(self) -> None:
        """
        Take an update step
        tau_{ij} <- rho * tau_{ij} + delta tau_{ij}
        """
        self.Tau = self.Tau + self.learning_rate * self._delta_tau()
        self.Tau[np.where(self.Tau < self.min_tau)] = self.min_tau

    def _advantage_local(self, **kwargs) -> float:
        """Advantage function of the form:
        1/C(x) - Avg(1/C(x))
        :returns: (float) calculated advantage

        """
        current_point = kwargs["current_point"]
        next_point = kwargs["next_point"]
        allow_list = kwargs["allow_list"]
        advantage = 1/self.distance_matrix[current_point, next_point] - np.average(1/self.distance_matrix[current_point, allow_list])
        return advantage

    def _advantage_path(self, **kwargs) -> float:
        """Advantage function of the form
        1/C(s_{t}) - Avg(1/C(s_{t-1}))
        :returns: (float) calculated average

        """
        pass


    def _ant_search(self, j) -> None:
        """Find a path for a single path."""
        self.Table[j, 0] = 0  # start at node 0
        ant_cost = []
        for k in range(self.n_dim - 1):
            current_point = self.Table[j, k]
            # get viable
            allow_list = self._get_candiates(set(self.Table[j, :k + 1]))
            # roulette selection
            tau_list = self.prob_matrix[current_point, allow_list]
            prob = tau_list / tau_list.sum()
            next_point = np.random.choice(allow_list, size=1, p=prob)[0]
            self.Table[j, k + 1] = next_point

            ant_cost.append((ant_cost[-1] if len(ant_cost) != 0 else 0) + 1/self.distance_matrix[current_point, next_point])

            # calculate log policy gradient
            # advantage = (ant_cost[-1] - self.value_matrix[k])
            advantage = self._advantage_local(current_point=current_point, next_point=next_point, allow_list=allow_list)
            self.Table_grad[current_point, next_point] += self.alpha *advantage / self.Tau[current_point, next_point]
            for point, prob_val in zip(allow_list, prob):
                self.Table_grad[current_point, point] -= self.alpha * advantage /self.Tau[current_point, point] * prob_val
        return ant_cost

    def run(self,
            max_iter=None,
            metric= ["best"]):
        if self.func is None or self.distance_matrix is None:
            raise ValueError("func and distance_matrix must be set to run ACA")

        self.set_metrics(metric)
        self.metrics = metric
        """Driver method for finding the opt to TSP problem."""
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.iteration = i
            self.prob_matrix = self._prob_rule()
            ant_costs = np.zeros((self.n_dim-1))
            for j in range(self.size_pop):
                antj_cost = self._ant_search(j)
                ant_costs += antj_cost
            self.Table_grad /= self.size_pop
            self.value_matrix = self.value_param * self.value_matrix + ant_costs/self.size_pop
            # get score of ants via given opt funciton
            self.y = np.array([self.func(i) for i in self.Table])
            self._get_best()

            self._phero_update()

            if self.save_file is not None and self.iteration % self._checkpoint_res == 0:
                self._save()
            if self._checkpoint_file is not None and self.iteration % self._checkpoint_res == 0:
                self._checkpoint()

        self.best_x = self.current_best_X[-1]
        self.best_y = self.current_best_Y[-1]
        return self.best_x, self.best_y

class PolicyGradient4ACA(ACA_TSP):
    """Implementation of ACA with prob ratio policy gradient update

    Attributes:
        distrtance_matrix = weighted adjacency matrix of the given TSP space.
        n_dim = number of nodes in the graph.
        func = function to optimize wrt.
        size_pop = number of ants.
        max_iter = maximum number of iterations.
        alpha = pheromone exponent.
        beta = bias exponent.
        rho = pheromone evaporation rate.
        min_tau = minimum pheromone value.
        prob_bias = probability bias rule.
        lamb = branching factor.
        save_file = path to save learned pheromone table.
        _checkpoint_file = path to save learned pheromone table.
        _checkpoint_res = checkpoint resolution.
        learning_rate = learning rate for the gradient update.
    """
    def __init__(self, func,
                 distance_matrix,
                 params: dict = {}) -> None:
        """Class specific params."""
        super().__init__(func, distance_matrix, params=params)
        self._name_ = "Policy Ratio"
        self.learning_rate = params.get("learning_rate", 1)
        self.Table_grad = np.zeros((self.n_dim, self.n_dim))
        self._Tau_last_gen = self._prob_rule()
        self.value_matrix = np.zeros((self.n_dim-1))
        self.value_param = params.get("value_param", 0.7)

    def _delta_tau(self) -> np.ndarray:
        """Calculate the update rule."""
        return self.Table_grad


    def _phero_update(self) -> None:
        """
        Take an update step
        tau_{ij} <- rho * tau_{ij} + delta tau_{ij}
        """

        self.Tau = self.Tau + self.learning_rate * self._delta_tau()
        self.Tau[np.where(self.Tau < self.min_tau)] = self.min_tau

    def _ant_search(self, j) -> None:
        """Find a path for a single path."""
        self.Table[j, 0] = 0  # start at node 0
        ant_cost = []
        for k in range(self.n_dim - 1):
            current_point = self.Table[j, k]
            # get viable
            allow_list = self._get_candiates(set(self.Table[j, :k + 1]))
            # roulette selection
            tau_list = self.prob_matrix[current_point, allow_list]
            prob = tau_list / tau_list.sum()
            tau_list_old = self._Tau_last_gen[current_point, allow_list]
            prob_old = tau_list_old / tau_list_old.sum()

            next_point = np.random.choice(allow_list, size=1, p=prob)[0]
            self.Table[j, k + 1] = next_point

            ant_cost.append((ant_cost[-1] if len(ant_cost) != 0 else 0) + 1/self.distance_matrix[current_point, next_point])

            # calculate log policy gradient
            advantage = 1/self.distance_matrix[current_point, next_point] - np.average(1/self.distance_matrix[current_point, allow_list])
            # advantage = (ant_cost[-1] - self.value_matrix[k])

            prob_chosen = self.prob_matrix[current_point, next_point]/tau_list.sum()
            prob_chosen_old = self._Tau_last_gen[current_point, next_point]/tau_list.sum()
            self.Table_grad[current_point, next_point] += self.alpha *advantage * (prob_chosen/prob_chosen_old)
            for point, prob_val in zip(allow_list, prob):

                self.Table_grad[current_point, point] -= self.alpha * advantage * (prob_chosen/prob_chosen_old) * prob_val
        return ant_cost

    def run(self,
            max_iter=None,
            metric= ["best"]):
        if self.func is None or self.distance_matrix is None:
            raise ValueError("func and distance_matrix must be set to run ACA")

        self.set_metrics(metric)
        self.metrics = metric
        """Driver method for finding the opt to TSP problem."""
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.iteration = i
            self.prob_matrix = self._prob_rule()
            ant_costs = np.zeros((self.n_dim-1))
            for j in range(self.size_pop):
                antj_cost = self._ant_search(j)
                ant_costs += antj_cost
            self.Table_grad /= self.size_pop
            self.value_matrix = self.value_param * self.value_matrix + ant_costs/self.size_pop
            # get score of ants via given opt funciton
            self.y = np.array([self.func(i) for i in self.Table])
            self._get_best()

            self._phero_update()

            if self.save_file is not None and self.iteration % self._checkpoint_res == 0:
                self._save()
            if self._checkpoint_file is not None and self.iteration % self._checkpoint_res == 0:
                self._checkpoint()

        self.best_x = self.current_best_X[-1]
        self.best_y = self.current_best_Y[-1]
        return self.best_x, self.best_y

class PolicyGradient5ACA(ACA_TSP):
    """Implementation of ACA with prob ratio policy gradient update with clip (this is PPOACA)

    Attributes:
        distrtance_matrix = weighted adjacency matrix of the given TSP space.
        n_dim = number of nodes in the graph.
        func = function to optimize wrt.
        size_pop = number of ants.
        max_iter = maximum number of iterations.
        alpha = pheromone exponent.
        beta = bias exponent.
        rho = pheromone evaporation rate.
        min_tau = minimum pheromone value.
        prob_bias = probability bias rule.
        epsilon = clipping value.
        lamb = branching factor.
        save_file = path to save learned pheromone table.
        _checkpoint_file = path to save learned pheromone table.
        _checkpoint_res = checkpoint resolution.
        learning_rate = learning rate for the gradient update.
    """
    def __init__(self, func,
                 distance_matrix,
                 params: dict = {}) -> None:
        """Class specific params."""
        super().__init__(func, distance_matrix, params=params)
        self._name_ = "Policy Ratio Clip"
        self.learning_rate = params.get("learning_rate", 100)
        self.epsilon = params.get("epsilon", 0.1)
        self.Table_grad = np.zeros((self.n_dim, self.n_dim))
        self._Tau_last_gen = self._prob_rule()
        self.value_matrix = np.zeros((self.n_dim-1))
        self.value_param = params.get("value_param", 0.7)

    def _delta_tau(self) -> np.ndarray:
        """Calculate the update rule."""
        return self.Table_grad


    def _phero_update(self) -> None:
        """
        Take an update step
        tau_{ij} <- rho * tau_{ij} + delta tau_{ij}
        """
        self.Tau = self.Tau + self.learning_rate * self._delta_tau()
        self.Tau[np.where(self.Tau < self.min_tau)] = self.min_tau

    def _ant_search(self, j) -> None:
        """Find a path for a single path."""
        self.Table[j, 0] = 0  # start at node 0
        ant_cost = []
        for k in range(self.n_dim - 1):
            current_point = self.Table[j, k]
            # get viable
            allow_list = self._get_candiates(set(self.Table[j, :k + 1]))
            # roulette selection
            tau_list = self.prob_matrix[current_point, allow_list]
            prob = tau_list / tau_list.sum()
            tau_list_old = self._Tau_last_gen[current_point, allow_list]
            prob_old = tau_list_old / tau_list_old.sum()

            next_point = np.random.choice(allow_list, size=1, p=prob)[0]
            self.Table[j, k + 1] = next_point

            ant_cost.append((ant_cost[-1] if len(ant_cost) != 0 else 0) + 1/self.distance_matrix[current_point, next_point])

            # calculate log policy gradient
            advantage = 1/self.distance_matrix[current_point, next_point] - np.average(1/self.distance_matrix[current_point, allow_list])
            # advantage = (ant_cost[-1] - self.value_matrix[k])

            prob_chosen = self.prob_matrix[current_point, next_point]/tau_list.sum()
            prob_chosen_old = self._Tau_last_gen[current_point, next_point]/tau_list.sum()
            ratio = prob_chosen/prob_chosen_old
            r = np.zeros((len(allow_list)))
            if 1-self.epsilon < ratio and ratio < 1+self.epsilon:
                for tau_val, prob_val, index in zip(self.Tau[current_point, allow_list], prob, range(len(allow_list))):
                    r[index] = -1 * self.alpha * ratio * prob_val / tau_val
                    if allow_list[index] == next_point: r[index] += self.alpha * ratio / tau_val
            else: r = np.zeros((len(allow_list)))

            self.Table_grad[current_point, allow_list] += advantage * r
        return ant_cost

    def run(self,
            max_iter=None,
            metric= ["best"]):
        if self.func is None or self.distance_matrix is None:
            raise ValueError("func and distance_matrix must be set to run ACA")

        self.set_metrics(metric)
        self.metrics = metric
        """Driver method for finding the opt to TSP problem."""
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.iteration = i
            self.prob_matrix = self._prob_rule()
            ant_costs = np.zeros((self.n_dim-1))
            for j in range(self.size_pop):
                antj_cost = self._ant_search(j)
                ant_costs += antj_cost
            self.Table_grad /= self.size_pop
            self.value_matrix = self.value_param * self.value_matrix + ant_costs/self.size_pop
            # get score of ants via given opt funciton
            self.y = np.array([self.func(i) for i in self.Table])
            self._get_best()

            self._phero_update()

            if self.save_file is not None and self.iteration % self._checkpoint_res == 0:
                self._save()
            if self._checkpoint_file is not None and self.iteration % self._checkpoint_res == 0:
                self._checkpoint()

        self.best_x = self.current_best_X[-1]
        self.best_y = self.current_best_Y[-1]
        return self.best_x, self.best_y


if __name__ == "__main__":
    plot = True
    size = 100
    runs = 1
    iterations = 500
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))
    # distance_matrix[np.where(distance_matrix == 0)] = 1e13

    def cal_total_distance(routine):
        size = len(routine)
        return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                    for i in range(size)])


    print("Running ACA")
    ACA_runs = []

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = PolicyGradient3ACA(func=cal_total_distance,
                      distance_matrix=distance_matrix,
                      params={"max_iter": iterations,
                              "save_file": save_file,
                              "checkpoint_res": 1,
                              "rho": 0.5,
                              "alpha": 1,
                              "beta": 1,
                              "pop_size": 10,
                              "bias": "inv_weight",
                              "learning_rate": 100,
                              "value_param": 0.1,
                              })
        skaco_sol, skaco_cost = aca.run(metric = ["branching_factor"])
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(aca.generation_best_Y, label=aca._name_)
        plt.legend()
        plt.show()

    # params, tau_table = aca._load("ACO_run_0.txt")
    # print(tau_table)

    G = nx.from_numpy_array(distance_matrix, create_using=nx.DiGraph)
    approx = nx.approximation.simulated_annealing_tsp(G, "greedy", source=0)


    print(cal_total_distance(approx))

    pass
