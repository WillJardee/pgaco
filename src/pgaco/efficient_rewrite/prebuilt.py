from numpy.typing import NDArray

from pgaco.efficient_rewrite import Aco

class As(Aco):
    def __init__(self,
                 graph: NDArray, *,
                 alpha: float = 1,
                 beta: float = 2,
                 rho: float = 0.9,
                 pop_size: int = 10,
                 greedy_epsilon: float = 0.1,
                 maxmin: bool = False,
                 maxmin_adaptive: bool = True,
                 buffer_size: int = 10,
                 buffer_rule: str = "evict",
                 kernel: str = "identity",
                 adv_func: str = "reward_path",
                 importance_sampling: bool = False,
                 seed: int | None = None,
                 random_pool_size: int | None = None,
                 gradient: str = "AS"
                 ):
        self.name = "AS"
        super().__init__(graph,
                         alpha=alpha,
                         beta=beta,
                         rho=rho,
                         pop_size=pop_size,
                         greedy_epsilon=greedy_epsilon,
                         maxmin=maxmin,
                         maxmin_adaptive=maxmin_adaptive,
                         buffer_size=buffer_size,
                         buffer_rule=buffer_rule,
                         kernel=kernel,
                         adv_func=adv_func,
                         importance_sampling=importance_sampling,
                         seed=seed,
                         random_pool_size=random_pool_size,
                         gradient=gradient
                         )
        self.name = "AS"

class Acosga(Aco):
    def __init__(self,
                 graph: NDArray, *,
                 alpha: float = 1,
                 beta: float = 2,
                 rho: float = 50,
                 pop_size: int = 10,
                 greedy_epsilon: float = 0.1,
                 # clip_epsilon: float = 0.2,
                 # learning_rate: float = 0.1,
                 maxmin: bool = True,
                 maxmin_adaptive: bool = True,
                 buffer_size: int | None = None,
                 buffer_rule: str = "evict",
                 kernel: str = "identity",
                 adv_func: str = "reward_path",
                 importance_sampling: bool = False,
                 seed: int | None = None,
                 random_pool_size: int | None = None,
                 gradient: str = "acosga"
                 ):
        self.name = "ACOSGA"
        super().__init__(graph,
                         alpha=alpha,
                         beta=beta,
                         rho=rho,
                         pop_size=pop_size,
                         greedy_epsilon=greedy_epsilon,
                         # clip_epsilon=clip_epsilon,
                         # learning_rate=learning_rate,
                         maxmin=maxmin,
                         maxmin_adaptive=maxmin_adaptive,
                         buffer_size=buffer_size,
                         buffer_rule=buffer_rule,
                         kernel=kernel,
                         adv_func=adv_func,
                         importance_sampling=importance_sampling,
                         seed=seed,
                         random_pool_size=random_pool_size,
                         gradient=gradient
                         )
        self.name = "ACOSGA"

class Adaco(Aco):
    def __init__(self,
                 graph: NDArray, *,
                 alpha: float = 1,
                 beta: float = 2,
                 rho: float = 50,
                 pop_size: int = 10,
                 greedy_epsilon: float = 0.1,
                 # clip_epsilon: float = 0.2,
                 learning_rate: float = 0.1,
                 maxmin: bool = True,
                 maxmin_adaptive: bool = True,
                 buffer_size: int | None = None,
                 buffer_rule: str = "evict",
                 kernel: str = "identity",
                 adv_func: str = "reward_path",
                 importance_sampling: bool = False,
                 seed: int | None = None,
                 random_pool_size: int | None = None,
                 gradient: str = "adaco"
                 ):
        self.name = "ADACO"
        super().__init__(graph,
                         alpha=alpha,
                         beta=beta,
                         rho=rho,
                         pop_size=pop_size,
                         greedy_epsilon=greedy_epsilon,
                         # clip_epsilon=clip_epsilon,
                         learning_rate=learning_rate,
                         maxmin=maxmin,
                         maxmin_adaptive=maxmin_adaptive,
                         buffer_size=buffer_size,
                         buffer_rule=buffer_rule,
                         kernel=kernel,
                         adv_func=adv_func,
                         importance_sampling=importance_sampling,
                         seed=seed,
                         random_pool_size=random_pool_size,
                         gradient=gradient
                         )
        self.name = "ADACO"

class Pgaco(Aco):
    def __init__(self,
                 graph: NDArray, *,
                 alpha: float = 1,
                 beta: float = 0,
                 rho: float = 50,
                 pop_size: int = 10,
                 greedy_epsilon: float = 0.1,
                 # clip_epsilon: float = 1,
                 # learning_rate: float = 0.1,
                 maxmin: bool = False,
                 maxmin_adaptive: bool = False,
                 buffer_size: int | None = 35,
                 buffer_rule: str = "elite",
                 kernel: str = "exp",
                 adv_func: str = "reward_path",
                 importance_sampling: bool = True,
                 seed: int | None = None,
                 random_pool_size: int | None = None,
                 gradient: str = "pg"
                 ):
        self.name = "PGACO"
        super().__init__(graph,
                         alpha=alpha,
                         beta=beta,
                         rho=rho,
                         pop_size=pop_size,
                         greedy_epsilon=greedy_epsilon,
                         # clip_epsilon=clip_epsilon,
                         # learning_rate=learning_rate,
                         maxmin=maxmin,
                         maxmin_adaptive=maxmin_adaptive,
                         buffer_size=buffer_size,
                         buffer_rule=buffer_rule,
                         kernel=kernel,
                         adv_func=adv_func,
                         importance_sampling=importance_sampling,
                         seed=seed,
                         random_pool_size=random_pool_size,
                         gradient=gradient
                         )
        self.name = "PGACO"

class Ppoaco(Aco):
    def __init__(self,
                 graph: NDArray, *,
                 alpha: float = 1,
                 beta: float = 0,
                 rho: float = 50,
                 pop_size: int = 10,
                 greedy_epsilon: float = 0.1,
                 clip_epsilon: float = 0.2,
                 # learning_rate: float = 0.1,
                 maxmin: bool = False,
                 maxmin_adaptive: bool = False,
                 buffer_size: int | None = 35,
                 buffer_rule: str = "elite",
                 kernel: str = "exp",
                 adv_func: str = "reward_path",
                 importance_sampling: bool = True,
                 seed: int | None = None,
                 random_pool_size: int | None = None,
                 gradient: str = "ppo"
                 ):
        self.name = "PPOACO"
        super().__init__(graph,
                         alpha=alpha,
                         beta=beta,
                         rho=rho,
                         pop_size=pop_size,
                         greedy_epsilon=greedy_epsilon,
                         clip_epsilon=clip_epsilon,
                         # learning_rate=learning_rate,
                         maxmin=maxmin,
                         maxmin_adaptive=maxmin_adaptive,
                         buffer_size=buffer_size,
                         buffer_rule=buffer_rule,
                         kernel=kernel,
                         adv_func=adv_func,
                         importance_sampling=importance_sampling,
                         seed=seed,
                         random_pool_size=random_pool_size,
                         gradient=gradient
                         )
        self.name = "PPOACO"
