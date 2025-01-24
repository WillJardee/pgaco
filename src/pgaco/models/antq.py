"""
Adaptive Gradient ACO implemented in <https://doi.org/10.1016/B978-1-55860-377-6.50039-6>.

Classes:
    ANTQ: ACO with a Q-learning update rule.

"""


from typing import Callable, Iterable
import numpy as np
from scipy.sparse import dok_matrix
from pgaco.models import ACO, path_len


class ANTQ(ACO):
    """Implementation of ACO with log policy gradient update.

    Attributes
    ----------
        See parent's documentation
        learning_rate = learning rate for the gradient update.
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 func: Callable[[np.ndarray, Iterable], float] = path_len,
                 *,
                 discount_factor: float = 0.4,
                 off_policy: bool = True,
                 **kwargs) -> None:
        """Class specific params."""
        super().__init__(distance_matrix, func, **kwargs)
        self._discount_factor = discount_factor
        self._off_policy = off_policy
        self._name_ = "ANT-Q" if self._off_policy else "ANT-SARSA"
        self._running_grad = np.zeros(self._heuristic_table.shape)
        self._name_ = "ANT-Q"

    @property
    def _discount_factor(self):
        return self.__discount_factor

    @_discount_factor.setter
    def _discount_factor(self, discount_factor):
        assert self._between(discount_factor, lower=0, upper=1, inclusive=True)
        self.__discount_factor = float(discount_factor)

    @property
    def _off_policy(self):
        return self.__off_policy

    @_off_policy.setter
    def _off_policy(self, off_policy):
        assert isinstance(off_policy, bool)
        self.__off_policy = int(off_policy)

    def _gradient(self, solution, cost):
        """Calculate the gradient for a single example."""
        sol_len = len(solution)
        # add 1/(path len) to each edge
        # grad = np.zeros(self._heuristic_table.shape)
        grad = dok_matrix(self._heuristic_table.shape)
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

    # def _gradient_update(self) -> None:
    #     """Take an gradient step."""
    #     # tot_grad = np.zeros(self._heuristic_table.shape)
    #     # for solution, cost in zip(self._replay_buffer, self._replay_buffer_fit):
    #     #     tot_grad += self._gradient(solution, cost)
    #     # tot_grad = tot_grad/self._replay_size
    #     tot_grad = np.sum(self._replay_buffer_grads)
    #     tot_grad = tot_grad/self._replay_size
    #
    #
    #     self._running_grad = self._running_grad/(self._size_pop)
    #
    #     self._heuristic_table = (1-self._evap_rate) * self._heuristic_table + self._evap_rate * (tot_grad + self._discount_factor * self._running_grad)
    #     # Notice that there is the missing max term here, that is moved to the gradient
    #     self._minmax()


def run_model1(distance_matrix, seed):
    aco = ANTQ(distance_matrix,
                slim = False,
               seed          = seed)
    aco.run(max_iter=max_iter)
    return aco.generation_best_Y, aco.generation_policy_score, aco._name_

def run_model2(distance_matrix, seed):
    aco = ANTQ(distance_matrix,
               off_policy=False,
                slim = False,
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

    print("running ANT-Q")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model1, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="cyan", label=aco_name)
    plot(aco_policy_runs, color="blue", label=aco_name + " policy")

    print("running ANT-SARSA")
    aco_runs, aco_policy_runs, aco_name = parallel_runs(run_model2, runs, distance_matrix, seed = 42)
    plot(aco_runs, color="green", label=aco_name)
    plot(aco_policy_runs, color="lime", label=aco_name + " policy")

    plt.legend()
    plt.tight_layout()
    plt.show()

    pass
