"""
Adaptive Gradient ACO implemented in <https://doi.org/10.1016/B978-1-55860-377-6.50039-6>.

Classes:
    ANTQ: ACO with a Q-learning update rule.

"""


import numpy as np

from pgaco.model.ACO import ACO_TSP


class ANTQ(ACO_TSP):
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
        self.allowed_params = {"learning_rate", "decay_rate", "off_policy"}
        super().__init__(distance_matrix, **self._passkwargs(**kwargs))
        self._name_ = "ANT-Q"
        self._learning_rate     = kwargs.get("learning_rate", 0.01)
        self._discount_factor   = kwargs.get("discount_factor", 0.1)
        self._off_policy        = kwargs.get("off_policy", True)
        self._running_grad      = np.zeros(self._heuristic_table.shape)


    def _gradient(self, solution, cost) -> np.ndarray:
        """Calculate the gradient for a single example."""
        sol_len = len(solution)
        # add 1/(path len) to each edge
        grad = np.ones(self._heuristic_table.shape)
        for k in range(sol_len):
            n1, n2 = solution[(k)%sol_len], solution[(k+1)%sol_len]
            grad[n1, n2] += 1 / cost
            if k < sol_len - 1:
                allow_list = self._get_candiates(set(solution[:k+1])) # get accessible points
                if self._off_policy: # Q-learning if off-policy
                    self._running_grad[solution[k], solution[k+1]] += self._heuristic_table[solution[k], allow_list[self._heuristic_table[solution[k], allow_list].argmax()]]
                else: # SARSA is on-policy
                    self._running_grad[solution[k], solution[k+1]] += self._heuristic_table[solution[k], solution[k+1]]
        return grad

    def _gradient_update(self) -> None:
        """Take an gradient step."""
        tot_grad = np.zeros(self._heuristic_table.shape)
        for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
            tot_grad += self._gradient(solution, cost)
        tot_grad = tot_grad/self._replay_size

        self._running_grad = self._running_grad/(self._size_pop)

        self._heuristic_table = (1-self._learning_rate) * self._heuristic_table + self._learning_rate * (tot_grad + self._discount_factor * self._running_grad)
        # Notice that there is the missing max term here, that is moved to the gradient
        self._minmax()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    size = 100
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 100, size**2).reshape((size, size))

    print("Running ACO")
    ACA_runs = []
    aca = ANTQ(distance_matrix,
                  max_iter = iterations)

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = ANTQ(distance_matrix,
                   max_iter = iterations)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
