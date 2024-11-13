from pgaco.tuning.tuning_base import *
from pgaco.models import ADACO

model_name = "ADACO"

def model(trial) -> float:
    size_pop    = trial.suggest_int("size_pop", size_pop_down, size_pop_up, log=True)
    alpha       = trial.suggest_int("alpha", alpha_down, alpha_up)
    beta        = trial.suggest_int("beta", beta_down, beta_up)
    evap_rate   = trial.suggest_float("evap_rate", evap_rate_down, evap_rate_up)
    decay_rate  = trial.suggest_float("decay_rate", 0.001, 0.99)

    replay_size = trial.suggest_int("replay_size", replay_size_down, replay_size_up)

    aco = ADACO(graph,
                seed          =  seed,
                size_pop      =  size_pop,
                alpha         =  alpha,
                beta          =  beta,
                evap_rate     =  evap_rate,
                replay_size   =  replay_size,
                decay_rate    =  decay_rate)

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

