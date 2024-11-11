import numpy as np
from ACA import ACA_TSP


###############################################################################
class PolicyGradientACA(ACA_TSP):
    """ACA with policy gradient update function."""
    def __init__(self, func,
                 distance_matrix,
                 params: dict = {}) -> None:
        """Class specific params."""
        super().__init__(func, distance_matrix, params=params)
        self.gamma = params.get("gamma", 1)
        self.learning_rate = params.get("learning_rate", 1)
        self.advantage_type = params.get("advantage_type", "forward_backward")

    def _baseline_advantage(self) -> np.ndarray:
        adv = np.zeros((self.n_dim, self.n_dim))
        for j in range(self.size_pop):
            cost_list = []
            # Get the cost of each edge
            for k in range(self.n_dim - 1):
                cost_list.append(self.distance_matrix[k, k+1])
            # For each edge, do a bellman sum backward and forward
            for k in range(self.n_dim - 1):
                running_sum = 0
                for i, item in enumerate(cost_list[:k]):
                    running_sum += (item * self.gamma ** (k-i))
                for i, item in enumerate(cost_list[k+1:]):
                    running_sum += (item * self.gamma ** (i+1))
                running_sum += cost_list[k]
                adv[k, k+1] += 1/running_sum
                # adv[k, k+1] += 1/cost_list[k]**self.n_dim
                adv[k, k+1] -= 1/self.y[j]
            return adv

    def _fb_Q(self, forward: bool, backward: bool) -> np.ndarray:
        adv = np.zeros((self.n_dim, self.n_dim))
        for j in range(self.size_pop):
            cost_list = []
            # Get the cost of each edge
            for k in range(self.n_dim - 1):
                cost_list.append(self.distance_matrix[k, k+1])
            # For each edge, do a bellman sum backward and forward
            for k in range(self.n_dim - 1):
                running_sum = 0
                if backward:
                    for i, item in enumerate(cost_list[:k]):
                        running_sum += (item * self.gamma ** (k-i))
                if forward:
                    for i, item in enumerate(cost_list[k+1:]):
                        running_sum += (item * self.gamma ** (i+1))
                running_sum += cost_list[k]
                adv[k, k+1] += 1/running_sum
                # adv[k, k+1] -= 1/self.y[j]
            return adv

    def _advantage(self):
        match self.advantage_type:
            case "forward_backward":
                return self._fb_Q(forward=True, backward=True)
            case "forward":
                return self._fb_Q(forward=True, backward=False)
            case "backward":
                return self._fb_Q(forward=False, backward=True)
            case "baseline":
                return self._baseline_advantage()
            case _:
                raise ValueError(f"advantage_type: {self.advantage_type} is not a valid advantage method.")

    def _log_grad(self):
        running_sum = 0
        grad = np.zeros(self.Tau.shape)
        for ant, score in zip(self.Table, self.y):
            for k in range(self.n_dim - 1):
                h = self._prob_rule_node(ant[k], ant[:k])
                tau = self.Tau[k][k+1]
                running_sum += h/tau
                grad[k][k+1] -= 1/(tau)
        return self.alpha * (grad + running_sum)

    def _phero_update(self) -> None:
        log_grad = self._log_grad()
        adv = self._advantage()
        self.Tau += self.learning_rate * log_grad * adv


###############################################################################
class PolicyGradient2ACA(ACA_TSP):
    """ACA with policy gradient update function."""

    def __init__(self, func,
                 distance_matrix,
                 params: dict = {}) -> None:
        """Class specific params."""
        super().__init__(func, distance_matrix, params=params)
        self.gamma = params.get("gamma", 1)
        self.learning_rate = params.get("learning_rate", 1)
        self.advantage_type = params.get("advantage_type", "forward_backward")

    def _baseline_advantage(self) -> np.ndarray:
        adv = np.zeros((self.n_dim, self.n_dim))
        for j in range(self.size_pop):
            cost_list = []
            # Get the cost of each edge
            for k in range(self.n_dim - 1):
                cost_list.append(self.distance_matrix[k, k+1])
            # For each edge, do a bellman sum backward and forward
            for k in range(self.n_dim - 1):
                running_sum = 0
                for i, item in enumerate(cost_list[:k]):
                    running_sum += (item * self.gamma ** (k-i))
                for i, item in enumerate(cost_list[k+1:]):
                    running_sum += (item * self.gamma ** (i+1))
                running_sum += cost_list[k]
                adv[k, k+1] += 1/running_sum
                # adv[k, k+1] += 1/cost_list[k]**self.n_dim
                adv[k, k+1] -= 1/self.y[j]
            return adv

    def _fb_Q(self, forward: bool, backward: bool) -> np.ndarray:
        adv = np.zeros((self.n_dim, self.n_dim))
        for j in range(self.size_pop):
            cost_list = []
            # Get the cost of each edge
            for k in range(self.n_dim - 1):
                cost_list.append(self.distance_matrix[k, k+1])
            # For each edge, do a bellman sum backward and forward
            for k in range(self.n_dim - 1):
                running_sum = 0
                if backward:
                    for i, item in enumerate(cost_list[:k]):
                        running_sum += (item * self.gamma ** (k-i))
                if forward:
                    for i, item in enumerate(cost_list[k+1:]):
                        running_sum += (item * self.gamma ** (i+1))
                running_sum += cost_list[k]
                adv[k, k+1] += 1/running_sum
                # adv[k, k+1] -= 1/self.y[j]
            return adv

    def _advantage(self):
        match self.advantage_type:
            case "forward_backward":
                return self._fb_Q(forward=True, backward=True)
            case "forward":
                return self._fb_Q(forward=True, backward=False)
            case "backward":
                return self._fb_Q(forward=False, backward=True)
            case "baseline":
                return self._baseline_advantage()
            case _:
                raise ValueError(f"advantage_type: {self.advantage_type} is not a valid advantage method.")

    def _prob_rule_node(self, node, taboo_list) -> float:
        probs = np.zeros(self.n_dim)
        allow_list = self._get_candiates(taboo_set=taboo_list)
        probs[allow_list] = self.prob_matrix[node][allow_list]
        probs /= probs.sum()
        return probs[node]

    def _log_grad(self):
        running_sum = 0
        grad = np.zeros(self.Tau.shape)
        for ant, score in zip(self.Table, self.y):
            for k in range(self.n_dim - 1):
                h = self._prob_rule_node(ant[k], ant[:k])
                tau = self.Tau[k][k+1]
                running_sum += h/tau
                grad[k][k+1] -= 1/(tau)
        return self.alpha * (grad + running_sum)

    def _phero_update(self) -> None:
        log_grad = self._log_grad()
        adv = self._advantage()
        self.Tau += self.learning_rate * log_grad * adv




