import optuna

class BurnInPruner(optuna.pruners.SuccessiveHalvingPruner):
    def __init__(self, n_burn_in_steps, **kwargs):
        super().__init__(**kwargs)
        self.n_burn_in_steps = n_burn_in_steps

    def prune(self, study, trial):
        step = trial.last_step
        if step < self.n_burn_in_steps:
            return False  # Ignore evaluations during burn-in
        return super().prune(study, trial)
