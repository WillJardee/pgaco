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
    # evap_rate, learning_rate, alpha, beta, min_tau = params
    evap_rate, alpha, beta, min_tau = params

    # Create your algorithm instance with the hyperparameters
    alg = ACO_TSP(distance_matrix,
                  evap_rate=evap_rate,
                  # learning_rate=learning_rate,
                  alpha=alpha,
                  beta=beta,
                  min_tau=min_tau)

    # Run the algorithm
    score, _ = alg.run()
    return float(score)

# Define the search space
search_space = [
    Real(0.01, 0.5, name='evap_rate'),  # Continuous parameter
    # Real(0.01, 10000, name='learning_rate'),  # Integer parameter
    Real(0, 5, name='alpha'),  # Integer parameter
    Real(0, 5, name='beta'),  # Integer parameter
    Real(0.00001, 100, name='min_tau'),  # Integer parameter
]
graph = 100
distance_matrix = get_graph(graph)

print(f"Tuning on {graph}.")

# Run Bayesian optimization
result = gp_minimize(objective, search_space, n_calls=52, random_state=42, n_jobs=-1)

# Get the best parameters and score
best_params = result.x
best_score = result.fun  # Negate back if you returned negative score

params = {space.name : best for space, best in zip(search_space, best_params)}

print(f"Best parameters: {params}")
print(f"Best score: {best_score}")
