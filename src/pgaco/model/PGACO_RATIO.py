import numpy as np
from tqdm import tqdm
import networkx as nx
import ast
import pickle
try: from .PGACO_LOG import PolicyGradient3ACA
except: from PGACO_LOG import PolicyGradient3ACA


class PolicyGradient4ACA(PolicyGradient3ACA):
    """Implementation of ACA with prob ratio policy gradient update"""
    def __init__(self, distance_matrix, **kwargs) -> None:
        """Class specific params."""
        super().__init__(distance_matrix, **kwargs)
        self._name_ = "Policy Ratio"
        self._prob_table_last_gen = self._prob_matrix

    def _gradient_add(self, current_point, next_point, allow_list, prob, advantage) -> None:
        """Add a value to the running gradient."""
        norm_term = self._prob_matrix[current_point, allow_list].sum()
        norm_term_old = self._prob_table_last_gen[current_point, allow_list].sum()
        prob_chosen = self._prob_matrix[current_point, next_point]/norm_term
        prob_chosen_old = self._prob_table_last_gen[current_point, next_point]/norm_term_old

        self._running_gradient[current_point, next_point] += self._alpha *advantage * (prob_chosen/prob_chosen_old)
        for point, prob_val in zip(allow_list, prob):
            self._running_gradient[current_point, point] -= self._alpha * advantage * (prob_chosen/prob_chosen_old) * prob_val

    def _gradient_update(self) -> None:
        super()._gradient_update()
        self._prob_table_last_gen = self._prob_matrix

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    size = 50
    runs = 5
    iterations = 100
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))

    print("Running GPACO with ratio update")
    ACA_runs = []

    for test in tqdm(range(runs)):
        save_file = f"ACO_run_{test}.txt"
        aca = PolicyGradient4ACA(distance_matrix,
                      max_iter = iterations,
                      save_file = save_file)
        skaco_cost, skaco_sol = aca.run()
        ACA_runs.append(skaco_cost)

    ACA_runs = np.array(ACA_runs)
    print(f"PGACO_ratio: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")

    plt.plot(aca.generation_best_Y)
    plt.show()

    pass
