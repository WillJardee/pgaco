"""
Adaptive Gradient ACO implemented in <https://doi.org/10.1016/j.swevo.2022.101046>.

Classes:
    ADACO: Policy Gradient ACO with log-gradient.

"""


import numpy as np

from pgaco.model.ACO import ACO_TSP


class ADACO(ACO_TSP):
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
                               "advantage_func", "annealing_factor",
                               "decay_rate"}
        super().__init__(distance_matrix, **self._passkwargs(**kwargs))
        self._name_ = "Adaptive ACO"
        self._decay_rate = kwargs.get("decay_rate", 0.95)
        self._decay_grad = np.zeros([self._dim, self._dim])
        self._delta_decay_grad = np.zeros([self._dim, self._dim])

    def _gradient_update(self) -> None:
        """Take an gradient step."""
        tot_grad = np.zeros(self._heuristic_table.shape)
        for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
            tot_grad += self._gradient(solution, cost)
        tot_grad = tot_grad/self._replay_size

        self._decay_grad = self._decay_rate * self._decay_grad + (1-self._decay_rate) * tot_grad**2
        epsilon = 1e-7
        hess = tot_grad * np.sqrt((self._delta_decay_grad + epsilon) / (self._decay_grad + epsilon) )
        self._heuristic_table = (1 - self._evap_rate) * self._heuristic_table - hess
        self._minmax()
        self._delta_decay_grad = self._decay_rate * self._delta_decay_grad + (1-self._decay_rate) * (hess * hess)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    size = 50
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))

    print("Running ADACO")
    ACA_runs = []
    aca = ADACO(distance_matrix,
                  max_iter = iterations)

    for _ in tqdm(range(runs)):
        aca = ADACO(distance_matrix,
                      max_iter = iterations)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
