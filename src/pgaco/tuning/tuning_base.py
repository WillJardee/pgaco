from datetime import datetime
from os.path import dirname
import sys

import optuna
import numpy as np

import pgaco

pruning_period  = 1    # period to check pruning at
n_trials        = 200
n_jobs          = 4     # -1 for as many as possible
seed            = 42
max_iter        = 100
default_graph   = "ali535.tsp"
model_name      = "ACO"
module_path     = dirname(pgaco.__spec__.origin)
save_dir        = f"{module_path}/results/tuning_params"
journal_name    = f"{model_name}.log"

gen_graph_down, gen_graph_up = 1, 100
size_pop_down, size_pop_up = 2, 100
alpha_down, alpha_up = 1, 10
beta_down, beta_up = 1, 10
evap_rate_down, evap_rate_up = 0.001, 0.99
replay_size_down, replay_size_up = 1, 500

def get_graph(name: str | int) -> np.ndarray:
    try: name = int(name)
    finally:
        if isinstance(name, int):
            size = int(name)
            dist_mat = np.random.randint(gen_graph_down, gen_graph_up, size**2).reshape((size, size))
            return  dist_mat.astype(np.float64)
        elif isinstance(name, str):
            from pgaco.utils.tsplib_reader import TSPGraph
            return TSPGraph(f"{name}").graph

def model(trial) -> float:
    raise NotImplementedError(f"Model needed for class you are testing.")

def main(model_to_run, model_name, graph_name: str| int | None = None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if isinstance(graph_name, str): graph_name = graph_name.strip(".tsp")
    journal_name = f"{graph_name}{model_name}.log"
    dandt = datetime.now()

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(f"{save_dir}/{journal_name}")
    )

    study = optuna.create_study(direction="minimize",
                                # pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                pruner=optuna.pruners.NopPruner(),
                                sampler=optuna.samplers.TPESampler(seed = seed,
                                                                   constant_liar=True),
                                # storage="sqlite:///db.sqlite3",
                                storage=storage,
                                study_name=f"{model_name}_{dandt.strftime('%Y-%m-%d_%H-%M-%S')}")
    if isinstance(graph_name, str): study.set_user_attr("graph", graph_name)
    elif isinstance(graph_name, int): study.set_user_attr("graph", str(graph_name))
    study.set_user_attr("contributor", "Will Jardee")
    study.set_user_attr("model", model_name)
    study.set_user_attr("seed", seed)
    print(study.pruner)
    # study.optimize(model_to_run, n_trials=n_trials, n_jobs=n_jobs)

    with open(f"{save_dir}/{model_name}_{dandt.strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as file_name:
        file_name.write('Best trial:')
        trial = study.best_trial
        file_name.write(f'  Value: {trial.value}\n')
        file_name.write('  Params: \n')
        for key, value in trial.params.items():
            file_name.write('    {}: {}\n'.format(key, value))

if __name__ == "__main__":
    graph_name = sys.argv[1] if len(sys.argv) > 1 else default_graph
    graph = get_graph(graph_name)
    main(model, model_name, graph_name=graph_name)
