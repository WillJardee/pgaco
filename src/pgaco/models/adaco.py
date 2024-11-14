"""
Adaptive Gradient ACO implemented in <https://doi.org/10.1016/j.swevo.2022.101046>.

Classes:
    ADACO: Policy Gradient ACO with log-gradient.

"""


import numpy as np
from typing import Callable, Iterable
from pgaco.models import ACOSGD, path_len

class ADACO(ACOSGD):
    """Implementation of ACO with log policy gradient update.

    Attributes
    ----------
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 func: Callable[[np.ndarray, Iterable], float] = path_len,
                 *,
                 decay_rate: float = 0.95,
                 **kwargs) -> None:
        self._name_ = "ADACO"
        super().__init__(distance_matrix, **kwargs)
        self._decay_rate = decay_rate
        self._decay_grad = np.zeros([self._dim, self._dim])
        self._delta_decay_grad = np.zeros([self._dim, self._dim])

    @property
    def _decay_rate(self):
        return self.__decay_rate

    @_decay_rate.setter
    def _decay_rate(self, decay_rate):
        assert self._between(decay_rate, lower=0, upper=1)
        self.__decay_rate = float(decay_rate)

    def _gradient_update(self) -> None:
        """Take an gradient step."""
        tot_grad = np.zeros(self._heuristic_table.shape)
        # for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
        #     tot_grad += self._gradient(solution, cost)
        # tot_grad = tot_grad/self._replay_size
        for grad in self._replay_buffer_grads:
            for coord, val in zip(grad.keys(), grad.values()):
                if coord == "sub_term":
                    continue
                tot_grad[coord[0], coord[1]] = val
            tot_grad -= grad["sub_term"]
        tot_grad = tot_grad/self._replay_size

        # for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
        #     tot_grad += self._gradient(solution, cost)
        # tot_grad = tot_grad/self._replay_size

        self._decay_grad = self._decay_rate * self._decay_grad + (1-self._decay_rate) * tot_grad**2
        epsilon = 1e-7
        hess = tot_grad * np.sqrt((self._delta_decay_grad + epsilon) / (self._decay_grad + epsilon) )
        self._heuristic_table = (1 - self._evap_rate) * self._heuristic_table - hess
        self._minmax()
        self._delta_decay_grad = self._decay_rate * self._delta_decay_grad + (1-self._decay_rate) * (hess * hess)


def run_model1(distance_matrix, seed):
    aco = ADACO(distance_matrix,
                size_pop      = 2,
                slim = False,
                seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_ + " w/ L2"

def run_model2(distance_matrix, seed):
    aco = ADACO(distance_matrix,
                size_pop      = 2,
                regularizer   = None,
                slim = False,
                seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pgaco.utils import get_graph, plot, parallel_runs
    size = 20
    runs = 5
    max_iter = 1500
    distance_matrix = get_graph(size)

    print("running ADACO w/ regularizer")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model1, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="cyan", label=aco_name)
    plot(aco_policy_runs, color="blue", label=aco_name + " policy")

    print("running ADACO")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model2, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="green", label=aco_name)
    plot(aco_policy_runs, color="lime", label=aco_name + " policy")

    plt.legend()
    plt.tight_layout()
    plt.show()

    pass
