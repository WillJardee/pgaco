import numpy as np

from ACA import ACA_TSP


class ACA_minmax(ACA_TSP):
    """ACA with minmax update function."""

    def __init__(self, func,
                 distance_matrix,
                 tau_min, tau_max,
                 params: dict = {}) -> None:
        """class specific params."""
        super().__init__(func, distance_matrix, params=params)
        self.tau_min = params.get("tau_min", 0.01*self.distance_matrix.min())
        self.tau_max = params.get("tau_max", 100*self.distance_matrix.max())
        self._name_ = "minmax aca"

    def _phero_update(self) -> None:
        """
        Take an update step
        """
        self.Tau = (1 - self.rho) * self.Tau + self.rho * self._delta_tau()
        self.Tau = np.minimum(self.Tau_max, np.maximum(self.Tau_min, self.Tau))


