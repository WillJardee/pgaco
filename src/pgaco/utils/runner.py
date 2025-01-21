import multiprocessing
from functools import partial

import numpy as np
import matplotlib.pyplot as plt


def plot(runs:np.ndarray, color:str, label:str, alpha: float = 0.3) -> None:
    runs = np.array(runs)
    aco_mean = np.mean(runs, axis = 0)
    aco_std = np.std(runs, axis=0)
    time_steps = np.arange(len(aco_mean))
    plt.plot(aco_mean, color=color, label=label)
    plt.fill_between(time_steps,
                     aco_mean - aco_std,
                     aco_mean + aco_std,
                     alpha=alpha, color=color)

def parallel_runs(alg, runs, distance_matrix, seed):
    with multiprocessing.Pool() as pool:
        run_func = partial(alg, distance_matrix)
        results = pool.map(run_func, range(seed, seed + runs))

    aco_runs = [result[0] for result in results]
    aco_policy_runs = [result[1] for result in results]
    aco_name = results[0][2]  # Assuming all runs have the same name
    return np.array(aco_runs), np.array(aco_policy_runs), str(aco_name)


