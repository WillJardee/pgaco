import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


from pgaco.efficient_rewrite.prebuilt import *

# Import Raytune
import os
os.environ["RAYON_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch

NUM_OPT_RUNS = 100


ITERS = 500
SEED = random.randint(0, int(1e6))
# SEED = 42

import tsplib95
# problem = tsplib95.load('../tsplib/berlin52.tsp')
graph_file = '../tsplib/att48.tsp'
opt = 10628
# plt.axhline(opt, color="black")
problem = tsplib95.load(graph_file)
G = problem.get_graph()
graph = nx.to_numpy_array(G)

print("ITERS: ", ITERS)
print("SEED: ", SEED)

beta = 1
# aco = Ppoaco(graph,
#              alpha=1,
#              rho=50,
#              pop_size=10,
#              greedy_epsilon=0.1,
#              clip_epsilon=0.2,
#              beta=beta
#              )
#

aco = Ppoaco(graph,
             alpha=1,
             # alpha=1,
             rho=10,
             pop_size=100,
             greedy_epsilon=0,
             clip_epsilon=0.2,
             buffer_size=100,
             beta=beta
             )
start_time = time.time()
aco.run(start=0, max_epochs=1)
AVGTIME = time.time() - start_time
AVGTIME *= 100

LOGPERIOD = int(np.log(ITERS))

def objective(config, final=False):
    aco = Ppoaco(graph,
                 alpha=config["alpha"],
                 # alpha=1,
                 rho=config["rho"],
                 pop_size=config["pop_size"],
                 greedy_epsilon=config["greedy_epsilon"],
                 clip_epsilon=config["clip_epsilon"],
                 buffer_size=config["buffer_size"],
                 beta=beta
                 )
    start_time = time.time()
    score_trace = []

    for i in range(ITERS):
        _, _ = aco.run(start=0, max_epochs=1)
        score = np.mean(aco.score_trace_heur[-1])
        score_trace.append(score)

        total_time = time.time() - start_time
        combined_metric = score * (1 + total_time/AVGTIME)  # Adjust weight as needed
        # tune.report({"score":score, "time":total_time, "combined_metric":combined_metric})
        if not final:
            if i % LOGPERIOD:
                tune.report({"combined_metric": combined_metric, "score": score})  # This sends the score to Tune.
        # tune.report(score=score, time=total_time, combined_metric=combined_metric)
        # tune.report(**{"score": score, "time": total_time, "combined_metric": combined_metric})

    if final:
        return score
    else:
        tune.report({"combined_metric": combined_metric, "score": score})  # This sends the score to Tune.
        return {"combined_metric": combined_metric, "score": score}

search_space ={
        "alpha": tune.uniform(0.5,1.5),
        "rho": tune.uniform(0.01, 100),
        "pop_size": tune.uniform(1,100),
        "greedy_epsilon": tune.uniform(0.001,0.5),
        "clip_epsilon": tune.uniform(0.001,0.5),
        "buffer_size": tune.uniform(1,200)
        }

tuner = tune.Tuner(
            objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=NUM_OPT_RUNS,  # Ensure multiple trials
                search_alg=BayesOptSearch(metric="combined_metric", mode="min")  # Bayesian Optimization
            ),
            #
            # run_config=tune.RunConfig(
            #     failure_config=tune.FailureConfig(max_failures=1),
            # )
        )
# class TrainableACO(tune.Trainable):
#
#     def __init__(self, config):
#         self.aco = Ppoaco(graph,
#                  alpha=config["alpha"],
#                  # alpha=1,
#                  rho=config["rho"],
#                  pop_size=config["pop_size"],
#                  greedy_epsilon=config["greedy_epsilon"],
#                  clip_epsilon=config["clip_epsilon"],
#                  buffer_size=config["buffer_size"],
#                  beta=beta
#                  )
#         self.start_time = time.time()
#
#     def step(self):
#         score_trace = []
#
#         for i in range(ITERS):
#             _, _ = self.aco.run(start=0, max_epochs=1)
#             score = np.mean(self.aco.score_trace_heur[-1])
#             # score_trace.append(score)
#             # tune.report({"score":score})
#
#         time_step = time.time() - self.start_time
#
#         # Define a new objective combining score and time
#         combined_metric = score + 0.01 * time_step   # Adjust weight as needed
#
#         # tune.report({"score":score, "time":total_time, "combined_metric":combined_metric})
#         return dict(score=score, combined_metric=combined_metric)
#
# tuner = tune.Tuner(
#             TrainableACO,
#             param_space=search_space,
#             tune_config=tune.TuneConfig(
#                 num_samples=NUM_OPT_RUNS,  # Ensure multiple trials
#                 search_alg=BayesOptSearch(metric="_metric", mode="min")  # Bayesian Optimization
#             ),
#             # run_config=tune.RunConfig(
#             #     failure_config=tune.FailureConfig(max_failures=1),
#             # )
#         )


results = tuner.fit()
best_result = results.get_best_result(metric="score", mode="min")  # Adjust mode if needed

print("Best Configuration:", best_result.config)
print("Best Score:", best_result)
print("Best Score (run):", objective(best_result.config, final=True))
# print("Best Score:", best_result.metrics["score"])
# print("Execution Time:", best_result.metrics["time"])
# print("Combined Metric:", best_result.metrics["combined_metric"])

# plt.plot(aco.gen_sol_score, label=f"{aco.name}: Best Solution")
# plt.plot(aco.gen_pol_score, label=f"{i}: Policy Solution")
# print(f"{aco.name} search best: ", score)
# print(f"{aco.name} policy + heur: ", [aco.greedy_solution(heur=True)[1] for _ in range(10)])
# print(f"{aco.name} policy:        ", [aco.greedy_solution(heur=False)[1] for _ in range(10)])
# print()
#

