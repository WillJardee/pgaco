from enum import Enum
import line_profiler
import numpy as np
from numpy.core.multiarray import where
from numpy.typing import NDArray


class RandomPool:
    def __init__(self,
                 pool_size: int,
                 seed: None | int):
        self.rng = np.random.default_rng(seed=seed)
        self.random_pool_size = pool_size
        self.random_pool = np.zeros(pool_size)
        self.random_pool_idx = 0

    def get_uni(self) -> float:
        # self.random_pool_idx = self.random_pool_idx + 1
        #
        # if self.random_pool_idx == self.random_pool_size:
        #     self.random_pool = self.rng.random(self.random_pool_size)
        #     self.random_pool_idx = 0
        #
        return self.rng.random()
        # return self.random_pool[self.random_pool_idx]

    def weighted_random(self, weights: NDArray):
        weights = weights.cumsum()
        u = self.get_uni()
        r =  int(weights.searchsorted(u * float(weights[-1])))
        return r

    def integer(self, n: int):
        return int(self.get_uni() * n)

    def rect(self, shape):
        return self.rng.random(size=shape)

class KERNEL(Enum):
    exponential = 1
    linear = 2
    quadratic = 3
    cubic = 4

class GRADNAME(Enum):
    AS = 1
    maxmin = 2
    acosga = 3
    adaco = 4
    pg = 5
    ppo = 6
    tr_ppo_rb = 7

class ADVNAME(Enum):
    adv_local = 1
    adv_path = 2
    quality = 3
    reward = 4
    reward_to_go = 5
    reward_baselined = 6
    reward_baselined_to_go = 7
    reward_decay = 8
    reward_path = 9


class Aco:
    # @line_profiler.profile
    def __init__(self,
                 graph: NDArray, *,
                 alpha: float = 1, # policy weight
                 beta: float = 2, # heur weight
                 rho: float = 0.9, # 1 - evap rate
                 pop_size: int = 10, # population size
                 greedy_epsilon: float = 0.1,
                 clip_epsilon: float = 0.2,
                 learning_rate: float = 0.1,
                 anneal_rate: float = 0.99,
                 maxmin: bool = False,
                 maxmin_adaptive: bool = True,
                 symmetric: bool = True,
                 buffer_size: int | None = 20,
                 buffer_draws: int | None = 35,
                 to_go_steps: int = 10,
                 reward_decay: float = 0.01,
                 annealing: float = 0.99,
                 buffer_rule: str = "elite",
                 kernel: str = "linear",
                 adv_func: str = "reward_path",
                 importance_sampling: bool = True,
                 seed: int | None = None,
                 random_pool_size: int | None = None,
                 gradient: str = "AS"
                 ):
        graph = graph
        self.true_distance_matrix = graph
        # self.distance_matrix = (graph - graph[np.where(graph > 0)].min() + 1e-6)/graph.std()
        self.distance_matrix = np.subtract(graph, np.min(graph, initial=0, where=~np.eye(graph.shape[0], dtype=bool)))
        # self.distance_matrix /= self.distance_matrix.std()
        self.distance_matrix += 1e-3
        # self.distance_matrix = graph**2
        # self.distance_matrix[np.where(self.distance_matrix > 0)] -= self.distance_matrix[np.where(self.distance_matrix > 0)].min(axis=0)
        # self.distance_matrix /= self.distance_matrix.std(axis=0)
        # self.distance_matrix **= 1/2
        self.n = self.distance_matrix.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.greedy_epsilon = greedy_epsilon
        self.clip_epsilon = clip_epsilon
        self.learning_rate = learning_rate
        self.pop_size = int(pop_size)
        self.grad_name = GRADNAME[gradient]
        self.anneal_rate = anneal_rate
        self.symmetric = symmetric
        self.to_go_steps = to_go_steps
        self.reward_decay = reward_decay
        self.anneal_rate = annealing
        self.baseline = np.mean(self.distance_matrix**(-1))


        self.epoch = 0
        self.gen_sol_score = []
        self.gen_pol_score = []
        self.gen_sol_score_policy = []

        self.rng = RandomPool(random_pool_size if random_pool_size is not None else self.n, seed)

        match kernel:
            case "identity":
                self.kernel = self.identity
            case "exp":
                self.kernel = self.expon
        self.kernel_name = kernel
        self.adv_func = ADVNAME[adv_func]

        self.heur_mat = ((self.distance_matrix)**-1).astype(np.float64)
        self.phero_mat = np.ones(self.distance_matrix.shape).astype(np.float64)
        # self.phero_mat = self.heur_mat.copy()
        if beta == 0:
            self.heur_mat = np.ones(self.distance_matrix.shape, dtype=np.float64)
        else:
            self.heur_mat **= beta
        # self.phero_mat = self.rng.rect(self.distance_matrix.shape).astype(np.float64)
        # self.phero_mat = np.zeros(self.distance_matrix.shape, dtype=np.float64) + 1e-6
        self.prob_base = self.kernel(self.phero_mat**self.alpha * self.heur_mat**self.beta)

        self.best_sol = np.zeros(self.n)
        self.best_score = np.inf
        self.best_policy_score = np.inf

        self.buffer_rule = buffer_rule
        self.buffer_size = int(pop_size) if buffer_size is None else int(buffer_size)
        self.buffer_draws = int(pop_size) if buffer_draws is None else int(buffer_draws)
        self.replay_buffer =[]
        self.replay_buffer_costs = []
        self.replay_buffer_adv = []
        self.replay_buffer_rating = []
        self.replay_buffer_is = []

        self.importance_sampling = importance_sampling

        self.exp_decay_grad = np.zeros(self.distance_matrix.shape)
        self.delta_grad = np.zeros(self.distance_matrix.shape)

        self.maxmin = maxmin
        self.maxmin_adaptive = maxmin_adaptive
        _, self.pheromax = self.greedy_solution()
        self.pheromin = self.pheromax / (2 * self.n)

        self.score_trace_no_heur = []
        self.score_trace_heur = []
        self.num_samples = 0

        self.name = "ACO"

    def expon(self, weight: NDArray):
        return np.exp(weight)
        # return np.exp(weight)


    def identity(self, weight: NDArray):
        return weight

    def greedy_solution(self, start: int | None = None, heur: bool = True):
        mask = np.ones(self.n, dtype=bool)
        trace = np.zeros(self.n + 1, dtype=int)
        valid = list(range(self.n))
        cost = 0

        node1 = start if start is not None else self.rng.integer(self.n)
        valid.pop(valid.index(node1))
        mask[node1] = False
        trace[0] = node1

        for idx in range(1, self.n):
            if heur:
                weights = self.prob_base[node1, mask]
            else:
                weights = self.phero_mat[node1, mask]
            node2_idx = weights.argmax()
            node2 = valid.pop(node2_idx)
            mask[node2] = False
            trace[idx] = node2
            cost += self.true_distance_matrix[node1, node2]
            node1 = node2

        trace[-1] = trace[0]
        cost += self.true_distance_matrix[trace[-2], trace[-1]]
        return trace, cost

    def get_solution_cost(self, trace):
        cost = 0
        node1 = trace[0]
        for node in trace[1:]:
            node2 = node
            cost += self.true_distance_matrix[node1, node2]
            node1 = node2
        cost += self.true_distance_matrix[trace[-2], trace[-1]]
        return cost


    @line_profiler.profile
    def find_solution(self, start = None):
        mask = np.ones(self.n, dtype=bool)
        trace = np.zeros(self.n + 1, dtype=int)
        adv = np.zeros(self.n, dtype=np.float64)
        reward_list = []
        valid = list(range(self.n))

        node1 = start if start is not None else self.rng.integer(self.n)
        valid.pop(valid.index(node1))
        mask[node1] = False
        trace[0] = node1

        trace_prob = np.ones(self.n, dtype=np.float64) if self.importance_sampling else None
        cost = 0

        self.baseline = 0
        for idx in range(1, self.n):
            weights = self.prob_base[node1, mask]
            if self.greedy_epsilon > 0 or (self.greedy_epsilon > self.rng.get_uni()):
                # node2_idx = weights.argmax()
                node2_idx = self.rng.integer(mask.sum())
            else:
                node2_idx = self.rng.weighted_random(weights)

            node2 = valid.pop(node2_idx)
            trace[idx] = node2
            cost += self.distance_matrix[node1, node2]

            if trace_prob is not None:
                # print(f"ant: {self.k}:", weights.sum(), weights.round(2))
                # print()
                # trace_prob[idx-1] = weights[node2_idx]/(weights.sum())
                trace_prob[idx-1] = weights[node2_idx]/(weights.sum()) * trace_prob[:idx-2]

            match self.adv_func:
                case ADVNAME.adv_local:
                    adv[idx-1] += weights[node2_idx] - weights.mean()
                case ADVNAME.quality:
                    raise NotImplementedError
                case ADVNAME.reward:
                    adv[idx-1] += 1/self.distance_matrix[node1, node2]
                case ADVNAME.reward_to_go:
                    reward_list.append(1/self.distance_matrix[node1, node2] * self.reward_decay**self.to_go_steps)
                    if self.to_go_steps == len(reward_list):
                        adv[idx-self.to_go_steps] += sum(reward_list)
                        reward_list.pop(0)
                    reward_list = [(1/self.reward_decay) * r for r in reward_list]
                case ADVNAME.reward_baselined:
                    r = 1/self.distance_matrix[node1, node2]
                    self.baseline = (self.baseline * self.num_samples + r)/(self.num_samples + 1)
                    adv[idx-1] += r - self.baseline
                    # adv[idx-1] += 1/self.distance_matrix[node1, node2] - (1/self.distance_matrix[node1]).mean()
                case ADVNAME.reward_baselined_to_go:
                    r = 1/self.distance_matrix[node1, node2]
                    self.baseline = (self.baseline * self.num_samples + r)/(self.num_samples + 1)

                    a = r - self.baseline
                    reward_list.append(a * self.reward_decay**self.to_go_steps)
                    if self.to_go_steps == len(reward_list):
                        adv[idx-self.to_go_steps] += sum(reward_list)
                        reward_list.pop(0)
                    reward_list = [(1/self.reward_decay) * r for r in reward_list]
                case ADVNAME.reward_decay:
                    raise NotImplementedError

            node1 = node2
            mask[node1] = False
            self.num_samples += 1

        if trace_prob is not None:
            trace_prob[-1] = 1
        trace[-1] = trace[0]
        cost += self.distance_matrix[trace[-2], trace[-1]]

        match self.adv_func:
            case ADVNAME.adv_local:
                adv[-1] += 0
            case ADVNAME.adv_path:
                adv += np.ones(len(adv)) * 1/cost
            case ADVNAME.reward_path:
                adv += np.ones(len(adv)) * 1/cost
            case ADVNAME.reward_to_go:
                reward_list.append(1/self.distance_matrix[trace[-2], trace[-1]] * self.reward_decay**self.to_go_steps)
                reward_list.pop(0)
                reward_list = [(1/self.reward_decay) * r for r in reward_list]
                for i in range(self.to_go_steps-1):
                    adv[-i] += sum(reward_list)
                    reward_list.pop(0)
                    reward_list = [(1/self.reward_decay) * r for r in reward_list]
            case ADVNAME.reward_baselined_to_go:
                r = 1/self.distance_matrix[trace[-2], trace[-1]]
                self.baseline = (self.baseline * self.num_samples + r)/(self.num_samples + 1)

                a = r - self.baseline
                reward_list.append(a * self.reward_decay**self.to_go_steps)
                reward_list.pop(0)
                reward_list = [(1/self.reward_decay) * r for r in reward_list]
                for i in range(self.to_go_steps-1):
                    adv[self.to_go_steps-i] += sum(reward_list)
                    reward_list.pop(0)
                    reward_list = [(1/self.reward_decay) * r for r in reward_list]

        return trace, cost, trace_prob, adv

    def update_buffer(self):
        match self.buffer_rule:
            case "elite":
                self.push_buffer(self.cost)
            case "evict":
                self.push_buffer(-1*self.epoch * np.ones(self.pop_size))

    def push_buffer(self, ratings):
        if len(self.replay_buffer_costs) == 0:
            self.replay_buffer = self.ant_sol
            self.replay_buffer_costs = self.cost
            self.replay_buffer_adv = self.adv
            self.replay_buffer_rating = ratings
            self.replay_buffer_probs = np.repeat(self.curr_prob_mat, len(self.cost))
            if self.importance_sampling:
                self.replay_buffer_is = self.imp_samp
            if len(self.replay_buffer_costs) > self.buffer_size:
                keep_idx = np.argpartition(self.replay_buffer_rating, self.buffer_size)[:self.buffer_size]
                self.replay_buffer = self.ant_sol[keep_idx]
                self.replay_buffer_costs = self.cost[keep_idx]
                self.replay_buffer_adv = self.adv[keep_idx]
                self.replay_buffer_rating = ratings[keep_idx]
                self.replay_buffer_probs = np.repeat(self.curr_prob_mat, len(self.cost))[keep_idx]
                if self.importance_sampling:
                    self.replay_buffer_is = self.imp_samp[keep_idx]

        elif len(self.replay_buffer_costs) + self.pop_size > self.buffer_size:
            keep_idx = np.argpartition(np.concatenate([self.replay_buffer_rating, ratings]), self.buffer_size)[:self.buffer_size]
            self.replay_buffer = np.concatenate([self.replay_buffer, self.ant_sol])[keep_idx]
            self.replay_buffer_costs = np.concatenate([self.replay_buffer_costs, self.cost])[keep_idx]
            self.replay_buffer_adv = np.concatenate([self.replay_buffer_adv, self.adv])[keep_idx]
            self.replay_buffer_rating = np.concatenate([self.replay_buffer_rating, ratings])[keep_idx]
            self.replay_buffer_probs = np.concatenate([self.replay_buffer_probs, np.repeat(self.curr_prob_mat, len(self.cost))])[keep_idx]
            if self.importance_sampling:
                self.replay_buffer_is = np.concatenate([self.replay_buffer_is, self.imp_samp])[keep_idx]
        else:
            self.replay_buffer = np.concatenate([self.replay_buffer, self.ant_sol])
            self.replay_buffer_costs = np.concatenate([self.replay_buffer_costs, self.cost])
            self.replay_buffer_adv = np.concatenate([self.replay_buffer_adv, self.adv])
            self.replay_buffer_rating = np.concatenate([self.replay_buffer_rating, ratings])
            self.replay_buffer_probs = np.concatenate([self.replay_buffer_probs, np.repeat(self.curr_prob_mat, len(self.cost))])
            if self.importance_sampling:
                self.replay_buffer_is = np.concatenate([self.replay_buffer_is, self.imp_samp])


    # @line_profiler.profile
    def gradient_update(self):
        grad = np.zeros(self.phero_mat.shape)
        prob_base_inv = self.phero_mat**(self.alpha - 1) * self.heur_mat
        grad_node = -1 * np.ones(self.n, dtype=np.float64)
        for trace, adv in zip(self.replay_buffer, self.replay_buffer_adv):
            node1 = trace[0]
            mask = np.ones(self.n, dtype=np.float64)
            mask[node1] = False
            for idx in range(1, self.n):
                # grad_node = 0 * grad_node - 1
                # grad_node -= 1
                node2 = trace[idx]
                # grad_node = -np.ones(self.n)
                grad_node.fill(-1)
                # prob_list = np.where(mask, self.prob_base[node1], 0)
                prob_list = self.prob_base[node1] * mask
                prob = prob_list[node2] / prob_list.sum()
                # grad_node = -1 * np.ones(self.n)
                grad_node[node2] += 1/prob
                # grad_node[node1] += 1/prob # to do symmetric, we need to pull prob_base[node2]
                grad_node *= adv[idx-1] * prob_base_inv[node1] * mask
                # grad_node *= prob_base_inv[node1] * mask

                mask[node2] = False

                grad[node1] += grad_node
                # grad[node2] -= adv[idx-1] * grad_node
                node1 = node2
        grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad

    def gradient_update_is(self):
        grad = np.zeros(self.phero_mat.shape)
        prob_base_inv = self.phero_mat**(-1)
        for trace, l, q in zip(self.replay_buffer, self.replay_buffer_costs, self.replay_buffer_is):
            adv = 1/l
            node1 = trace[0]
            mask = np.ones(self.n, dtype=bool)
            mask[node1] = True
            for idx in range(1, self.n - 1):
                prob = self.prob_base[node1] * mask
                prob /= prob.sum()
                node2 = trace[idx]
                r = prob[node2]/q[idx]
                prob[node2] -= 1
                x = prob_base_inv[node1]
                val = -1.0 * r * (adv)
                prob *= x * val
                mask[node2] = False

                grad[node1] -= prob
                grad[node2] -= prob
                node1 = node2
        grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad

    def pg_update(self):
        grad = np.zeros(self.phero_mat.shape)
        grad_base = self.phero_mat**(self.alpha - 1) * self.heur_mat
        for trace, l, q in zip(self.replay_buffer, self.replay_buffer_costs, self.replay_buffer_is):
            adv = 1/l
            node1 = trace[0]
            mask = np.ones(self.n, dtype=bool)
            mask[node1] = True
            for idx in range(1, self.n - 1):
                prob = self.prob_base[node1] * mask
                prob /= prob.sum()
                node2 = trace[idx]
                r = prob[node2]/q[idx]

                prob[node2] -= 1
                x = -1.0 * r * (adv) * prob[node2] * grad_base[node1] * mask
                mask[node2] = False

                grad[node1] -= x
                grad[node2] -= x
                node1 = node2
        grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad

    @line_profiler.profile
    def pg_log_update(self):
        grad = np.zeros(self.phero_mat.shape)
        prob_base_inv = self.phero_mat**(self.alpha - 1) * self.heur_mat
        grad_node = -1 * np.ones(self.n, dtype=np.float64)
        for trace, adv in zip(self.replay_buffer, self.replay_buffer_adv):
            node1 = trace[0]
            mask = np.ones(self.n, dtype=np.float64)
            mask[node1] = False
            for idx in range(1, self.n):
                # grad_node = 0 * grad_node - 1
                # grad_node -= 1
                node2 = trace[idx]
                # grad_node = -np.ones(self.n)
                # grad_node.fill(-1)
                # prob_list = np.where(mask, self.prob_base[node1], 0)
                grad_node = -1 * self.prob_base[node1]
                grad_node /= grad_node.sum()
                # grad_node = -1 * np.ones(self.n)
                grad_node[node2] += 1
                # grad_node[node1] += 1/prob # to do symmetric, we need to pull prob_base[node2]
                grad_node *= adv[idx-1] * prob_base_inv[node1] * mask
                # grad_node *= prob_base_inv[node1] * mask

                mask[node2] = False

                grad[node1] += grad_node
                # grad[node2] += grad_node
                node1 = node2
        grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad


        # grad = np.zeros(self.phero_mat.shape)
        # grad_base = self.phero_mat**(self.alpha - 1) * self.heur_mat
        # for trace, adv, q in zip(self.replay_buffer, self.replay_buffer_adv, self.replay_buffer_is):
        #     node1 = trace[0]
        #     mask = np.ones(self.n, dtype=bool)
        #     mask[node1] = True
        #     for idx in range(1, self.n - 1):
        #         prob = self.prob_base[node1] * mask
        #         prob /= prob.sum()
        #         node2 = trace[idx]
        #         r = prob[node2]/q[idx]
        #
        #         prob[node2] -= prob[node2]
        #         x = -1.0 * r * (adv) * grad_base[node1] * mask
        #         x *= self.alpha * (1/prob.sum())
        #         mask[node2] = False
        #
        #         grad[node1] -= x
        #         grad[node2] -= x
        #         node1 = node2
        # grad /= self.buffer_size
        # self.phero_mat +=  self.rho * grad

    def pg_kernel_log_update(self):
        grad = np.zeros(self.phero_mat.shape)
        grad_base = self.phero_mat**(self.alpha - 1) * self.heur_mat
        for trace, l, q in zip(self.replay_buffer, self.replay_buffer_costs, self.replay_buffer_is):
            adv = 1/l
            node1 = trace[0]
            mask = np.ones(self.n, dtype=bool)
            mask[node1] = True
            for idx in range(1, self.n - 1):
                prob = self.prob_base[node1] * mask
                prob /= prob.sum()
                node2 = trace[idx]
                r = prob[node2]/q[idx]

                prob[node2] -= 1
                x = prob[node2] * self.alpha * -1.0 * r * (adv) * grad_base[node1] * prob
                mask[node2] = False

                grad[node1] -= x
                grad[node2] -= x
                node1 = node2
        grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad

    def ppo_update(self):
        grad = np.zeros(self.phero_mat.shape)
        # prob_base_inv = self.phero_mat**(self.alpha - 1) * self.heur_mat
        # print("epoch: ", self.epoch)
        # print("sum:",self.phero_mat.sum(axis=1))
        # print()
        # print("pre:", self.phero_mat)
        norm_count = np.ones(self.phero_mat.shape)
        grad_node = -1 * np.ones(self.n, dtype=np.float64)
        for _ in range(self.buffer_draws):
            sampled = self.rng.integer(len(self.replay_buffer_adv))
            # sampled = self.rng.weighted_random(self.replay_buffer_costs)
            trace, adv, q = self.replay_buffer[sampled], self.replay_buffer_adv[sampled], self.replay_buffer_is[sampled]
            node1 = trace[0]
            mask = np.ones(self.n, dtype=np.float64)
            mask[node1] = 0
            for idx in range(1, self.n):
                node2 = trace[idx]
                a = adv[idx-1]

                grad_node = -1 * self.prob_base[node1] * mask
                z = grad_node.sum()
                grad_node /= z
                r = (grad_node[node2])/(q[idx])

                # clipping
                if (r < 1-self.clip_epsilon) and (a < 0):  continue
                elif (r > 1+self.clip_epsilon) and (a > 0): continue
                grad_node[node2] += 1
                grad_node *= a * r * self.alpha * self.heur_mat[node1] * mask

                grad[node1] += grad_node
                if self.symmetric: grad[node2] += grad_node

                norm_count[node1, node2] += 1
                mask[node2] = 0
                node1 = node2
        grad /= norm_count
        # grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad
        # self.rho *= self.anneal_rate
        np.nan_to_num(self.phero_mat, nan=0.0)
        # self.phero_mat -= 0.001 * self.rho * self.phero_mat
        # print("post:",  self.phero_mat)
        # np.subtract(self.phero_mat,
        #             # self.phero_mat.max(axis=1, initial=1e-10, where=~np.eye(self.n,dtype=bool)).reshape((-1,1)),
        #             self.phero_mat.mean(axis=1).reshape((-1,1)),
        #             # self.phero_mat.min(axis=1).reshape((-1,1)),
        #             out = self.phero_mat
        #             )
        # print("post:",  self.phero_mat)
        # np.divide(self.phero_mat,
        #           self.phero_mat.std(axis=1, where=~np.eye(self.n, dtype=bool)).reshape((-1,1)),
        #           # self.phero_mat.max(axis=1, initial=1e-10, where=~np.eye(self.n,dtype=bool)).reshape((-1,1)) -
        #           # self.phero_mat.min(axis=1, initial=1e10, where=~np.eye(self.n,dtype=bool)).reshape((-1,1)),
        #           out = self.phero_mat
        #           )
        # print("post:",  self.phero_mat)




    # def tr_ppo_rb_update(self):
    #     grad = np.zeros(self.phero_mat.shape)
    #     prob_base_inv = self.phero_mat**(-1)
    #     ratios = np.zeros((self.n + 1, self.buffer_size))
    #     for trace, l, q, k in zip(self.replay_buffer, self.replay_buffer_costs, self.replay_buffer_is, np.arange(self.buffer_size)):
    #         adv = 1/l
    #         node1 = trace[0]
    #         mask = np.ones(self.n, dtype=bool)
    #         mask[node1] = True
    #         for idx in range(1, self.n - 1):
    #             prob = self.prob_base[node1] * mask
    #             prob /= prob.sum()
    #             node2 = trace[idx]
    #             r = prob[node2]/q[idx]
    #             ratios[idx, k] = r
    #     kl_div = np.log(ratios).sum()
    #
    #
    #     for trace, l, q in zip(self.replay_buffer, self.replay_buffer_costs, self.replay_buffer_is):
    #         for idx in range(1, self.n - 1):
    #             prob = self.prob_base[node1] * mask
    #             prob /= prob.sum()
    #             node2 = trace[idx]
    #             r = prob[node2]/q[idx]
    #             # clipping
    #             if (r < 1-self.clip_epsilon) and (adv < 0):
    #                 continue
    #             elif (r > 1-self.clip_epsilon) and (adv > 0):
    #                 continue
    #             prob[node2] -= 1
    #             x = prob_base_inv[node1]
    #             val = -1.0 * r * (adv)
    #             prob *= x * val
    #             mask[node2] = False
    #
    #             grad[node1] -= prob
    #             grad[node2] -= prob
    #             node1 = node2
    #     grad /= self.buffer_size
    #     self.phero_mat +=  self.rho * grad


    def adaco_update(self):
        grad = np.zeros(self.phero_mat.shape)
        prob_base_inv = self.phero_mat**(-1)
        for trace, l in zip(self.replay_buffer, self.replay_buffer_costs):
            adv = 1/l
            node1 = trace[0]
            mask = np.ones(self.n, dtype=bool)
            mask[node1] = True
            for idx in range(1, self.n - 1):
                prob = self.prob_base[node1] * mask
                prob /= prob.sum() # this replaces prob.sum(), which is slower?
                node2 = trace[idx]
                prob[node2] -= 1
                x = prob_base_inv[node1]
                val = -1.0 * (adv)
                prob *= x * val
                mask[node2] = False

                grad[node1] -= prob
                grad[node2] -= prob
                node1 = node2
        grad /= self.buffer_size * self.rho
        self.exp_decay_grad = self.learning_rate * (self.exp_decay_grad) + (1-self.learning_rate) * grad * grad
        self.hessian = grad * ((self.delta_grad + 1e-7) / (self.exp_decay_grad + 1e-7))**(1/2)
        self.delta_grad *= self.learning_rate
        self.delta_grad += (1-self.learning_rate) * self.hessian * self.hessian

        self.phero_mat -= self.hessian

    def adaco_update_approx(self):
        grad = np.zeros(self.phero_mat.shape)
        for trace, l in zip(self.replay_buffer, self.replay_buffer_costs):
            node1 = trace[0]
            for i in range(1, self.n - 1):
                node2 = trace[i]
                grad[node1, node2] += 0.01/l
                grad[node2, node1] += 0.01/l
                node1 = node2
        grad /= self.buffer_size * self.rho
        self.exp_decay_grad = self.learning_rate * (self.exp_decay_grad) + (1-self.learning_rate) * grad * grad
        self.hessian = grad * ((self.delta_grad + 1e-7) / (self.exp_decay_grad + 1e-7))**(1/2)
        self.delta_grad *= self.learning_rate
        self.delta_grad += (1-self.learning_rate) * self.hessian * self.hessian

        self.phero_mat -= self.hessian

    def grad_update(self):
        grad = np.zeros(self.phero_mat.shape)
        for trace, l in zip(self.replay_buffer, self.replay_buffer_costs):
            node1 = trace[0]
            for i in range(1, self.n - 1):
                node2 = trace[i]
                grad[node1, node2] += 1/l
                grad[node2, node1] += 1/l
                node1 = node2
        grad /= self.buffer_size
        self.phero_mat *= self.rho
        self.phero_mat += (1-self.rho) * grad

    def prob_base_update(self):
        # if not self.maxmin:
        #     self.phero_mat = (self.phero_mat - self.phero_mat.min() + 1e-6)/self.phero_mat.std() + 1
        if self.kernel_name == "exp":
            power = self.alpha * (self.phero_mat) * self.heur_mat ** (1/self.alpha)
            self.prob_base = np.exp(power - power.max())
        elif self.kernel_name == "identity":
            self.prob_base = (self.phero_mat**self.alpha) * self.heur_mat
        else:
            raise NotImplementedError(f"There is no kernel named '{self.kernel_name}'")

    def maxmin_clip(self):
        self.phero_mat[np.where(self.phero_mat < self.pheromin)] = self.pheromin
        self.phero_mat[np.where(self.phero_mat > self.pheromax)] = self.pheromax

        if self.maxmin_adaptive:
            _, self.pheromax = self.greedy_solution()
            self.pheromin = self.pheromax / (2 * self.n)

    @line_profiler.profile
    def run(self, start = None, max_epochs: int = 100):
        if self.epoch == 0:
            self.ant_sol = np.zeros((self.pop_size, self.n + 1), dtype=int)
            self.adv = np.zeros((self.pop_size, self.n), dtype=np.float64)
            self.cost = np.zeros(self.pop_size)
            self.imp_samp = np.zeros((self.pop_size, self.n), dtype=np.float64)
            self.score_trace_no_heur.append([self.greedy_solution(heur=False)[1] for _ in range(self.n)])
            self.score_trace_heur.append([self.greedy_solution(heur=True)[1] for _ in range(self.n)])
            self.path_sum = 0
            self.path_average = 0

        for epoch in range(max_epochs):
            self.epoch = epoch
            self.prob_base_update()
            for k in range(self.pop_size):
                self.k = k
                self.ant_sol[k], self.cost[k], self.imp_samp[k], self.adv[k] = self.find_solution(start=start)
                c = self.get_solution_cost(self.ant_sol[k])
                if c < self.best_score:
                    self.best_score = c
                    self.best_sol = self.ant_sol[k]

            # self.path_sum += self.cost.sum()
            # self.path_average = self.path_sum / self.pop_size * (self.epoch + 1)
            self.path_sum = self.cost.sum()
            self.path_average = self.path_sum / self.pop_size * (self.epoch + 1)
            # self.cost -= self.path_average
            match self.adv_func:
                case ADVNAME.adv_path:
                    self.adv -= 1/self.path_average
            self.gen_sol_score.append(self.best_score)
            self.gen_pol_score.append(self.greedy_solution(start=start)[1])
            self.curr_prob_mat = self.prob_base / self.prob_base.sum(axis=1).reshape((-1,1))
            self.update_buffer()

            match self.grad_name:
                case GRADNAME.AS:
                    # if self.maxmin:
                    #     self.maxmin_clip()
                    self.grad_update()
                case GRADNAME.maxmin:
                    raise(NotImplementedError)
                case GRADNAME.adaco:
                    self.adaco_update()
                    # if self.maxmin:
                    #     self.maxmin_clip()
                case GRADNAME.acosga:
                    self.gradient_update()
                    # if self.maxmin:
                    #     self.maxmin_clip()
                case GRADNAME.pg:
                    if self.kernel_name == "identity":
                    # self.pg_update()
                        self.pg_log_update()
                    elif self.kernel_name == "exp":
                        self.pg_log_update()
                    # if self.maxmin:
                    #     self.maxmin_clip()
                case GRADNAME.ppo:
                    self.ppo_update()
                case GRADNAME.tr_ppo_rb:
                    raise(NotImplementedError)
            if self.maxmin:
                self.maxmin_clip()

            self.score_trace_no_heur.append([self.greedy_solution(heur=False)[1] for _ in range(self.n)])
            self.score_trace_heur.append([self.greedy_solution(heur=True)[1] for _ in range(self.n)])
            c = np.min(self.score_trace_no_heur[-1])
            if c < self.best_policy_score:
                self.best_policy_score = c
            self.gen_sol_score_policy.append(self.best_policy_score)

        return self.best_sol, self.best_score

