import time
import sys

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from model.PGACO_RATIO_CLIP import PolicyGradient5ACA

def get_graph(name):
    try:
        size = int(name)
        dist_mat = np.random.randint(1, 10, size**2).reshape((size, size))
        return  dist_mat.astype(np.float64)
    except (ValueError, TypeError):
        from utils.tsplib_reader import TSPGraph
        return TSPGraph(f"{name}").graph


# Define the objective function
def objective(params):
    # Unpack the hyperparameters
    evap_rate, learning_rate, annealing_factor, alpha, beta, epsilon = params

    # Create your algorithm instance with the hyperparameters
    alg = PolicyGradient5ACA(distance_matrix,
                             evap_rate=evap_rate,
                             learning_rate=learning_rate,
                             annealing_factor=annealing_factor,
                             epsilon=epsilon,
                             alpha=alpha,
                             beta=beta,
                             max_iter=500)

    # Run the algorithm
    score, _ = alg.run()
    return float(score)


# Define the search space
search_space = [
    Real(0.01, 0.9, name='evap_rate'),  # Continuous parameter
    Real(0.01, 10000, name='learning_rate'),
    Real(0.0001, 0.5, name='annealing_factor'),
    Real(0, 10, name='alpha'),
    Real(0, 10, name='beta'),
    Real(0.001, 0.5, name='epsilon'),
]
graph = sys.argv[1]
distance_matrix = get_graph(graph)

alg = PolicyGradient5ACA(distance_matrix)
print(f"Tuning algorithm: {alg._name_}.")
print(f"Tuning on {graph}.")

begin = time.time()

# Run Bayesian optimization
result = gp_minimize(objective, search_space, n_calls=100, random_state=42, n_jobs=2)

end = time.time()
print(f"Elapsed time: {end - begin} sec.")

# Get the best parameters and score
best_params = result.x
best_score = result.fun  # Negate back if you returned negative score

params = {space.name : best for space, best in zip(search_space, best_params)}

print(f"Best parameters: {params}")
print(f"Best score: {best_score}")
