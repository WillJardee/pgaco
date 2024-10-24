import numpy as np

from ACA import ACA_TSP


class ACA_Elite(ACA_TSP):
    """ACA with only updates with the maximum"""
    def __init__(self, func,
                 distance_matrix,
                 tau_min, tau_max,
                 params: dict = {}) -> none:
        """class specific params."""
        super().__init__(func, distance_matrix, params=params)
        self._name_ = "Elite ACA"

    def _delta_tau(self) -> np.ndarray:
        """Calculate the update rule."""
        delta_tau = np.zeros((self.n_dim, self.n_dim))
        x_best, y_best = self._get_best()
        for k in range(self.n_dim - 1):
            n1, n2 = x_best[k], x_best[k + 1]
            delta_tau[n1, n2] += 1 / y_best
        n1, n2 = x_best[self.n_dim - 1], x_best[0]
        delta_tau[n1, n2] += 1 / y_best
        return delta_tau