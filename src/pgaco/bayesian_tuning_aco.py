import time
import sys

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from model.ACO import ACO_TSP

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
    evap_rate, alpha, beta, replay_size, pop_size= params

    # Create your algorithm instance with the hyperparameters
    alg = ACO_TSP(distance_matrix,
                  evap_rate=evap_rate,
                  alpha=alpha,
                  beta=beta,
                  replay_size=replay_size,
                  size_pop=pop_size,
                  max_iter=500)

    # Run the algorithm
    score, _ = alg.run()
    return float(score)


# Define the search space
search_space = [
    Real(0.01, 0.9, name='evap_rate'),  # Continuous parameter
    Integer(0, 10, name='alpha'),
    Integer(0, 10, name='beta'),
    Categorical([10, 100, 1000], name='replay_size'),
    Categorical([2, 10, 20, 50, 100, 200, 500], name='pop_size')
]
graph = sys.argv[1]
distance_matrix = get_graph(graph)

alg = ACO_TSP(distance_matrix)
print(f"Tuning algorithm: {alg._name_}.")
print(f"Tuning on {graph}.")

begin = time.time()

# Run Bayesian optimization
result = gp_minimize(objective, search_space, n_calls=200, random_state=42, n_jobs=-1)

end = time.time()
print(f"Elapsed time: {end - begin} sec.")

# Get the best parameters and score
best_params = result.x
best_score = result.fun  # Negate back if you returned negative score

params = {space.name : best for space, best in zip(search_space, best_params)}

print(f"Best parameters: {params}")
print(f"Best score: {best_score}")
