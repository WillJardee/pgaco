import numpy as np
from tqdm import tqdm
from pgaco.model.ACO import ACO_TSP

class ADACO(ACO_TSP):
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
                               "advantage_func", "annealing_factor",
                               "decay_rate"}
        super().__init__(distance_matrix, **self._passkwargs(**kwargs))
        self._name_ = "Adaptive ACO"
        self._decay_rate = kwargs.get("decay_rate", 0.95)
        self._decay_grad = np.zeros([self._dim, self._dim])
        self._delta_decay_grad = np.zeros([self._dim, self._dim])

    def _gradient_update(self) -> None:
        """Take an gradient step"""
        grad = self._gradient()
        self._decay_grad = self._decay_rate * self._decay_grad + (1-self._decay_rate) * grad**2
        epsilon = 1e-7
        hess = grad * np.sqrt((self._delta_decay_grad + epsilon) / (self._decay_grad + epsilon) )
        self._heuristic_table = (1 - self._evap_rate) * self._heuristic_table - hess
        self._minmax()
        self._delta_decay_grad = self._decay_rate * self._delta_decay_grad + (1-self._decay_rate) * (hess * hess)

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    size = 50
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))

    print("Running ADACO")
    ACA_runs = []
    aca = ADACO(distance_matrix,
                  max_iter = iterations)

    for test in tqdm(range(runs)):
        aca = ADACO(distance_matrix,
                      max_iter = iterations)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
