from pgaco.tuning.tuning_base import *
from pgaco.models import ACOSGD

model_name = "ACOSGD"

def model(trial) -> float:
    size_pop        = trial.suggest_int("size_pop", size_pop_down, size_pop_up, log=True)
    alpha           = trial.suggest_int("alpha", alpha_down, alpha_up)
    beta            = trial.suggest_int("beta", beta_down, beta_up)
    evap_rate       = trial.suggest_float("evap_rate", 1e-4, 1e2, log=True)
    annealing_factor = trial.suggest_float("annealing_factor", 0.001, 0.99)

    replay_size = trial.suggest_int("replay_size", replay_size_down, replay_size_up)

    aco = ACOSGD(graph,
                seed          =   seed,
                size_pop      =   size_pop,
                alpha         =   alpha,
                beta          =   beta,
                evap_rate     =   evap_rate,
                annealing_factor =   annealing_factor,
                replay_size   =   replay_size)

    for i in range(max_iter // pruning_period):
        intermediate_score, _ = aco.take_step(steps=pruning_period)
        trial.report(intermediate_score, i)
        if trial.should_prune(): raise optuna.TrialPruned()
    score, _ = aco.take_step(steps=max_iter%pruning_period)
    trial.report(score, (max_iter // pruning_period) + 1)
    return score

if __name__ == "__main__":
    graph_name = sys.argv[1] if len(sys.argv) > 1 else default_graph
    graph = get_graph(graph_name)
    main(model, model_name, graph_name=graph_name)

