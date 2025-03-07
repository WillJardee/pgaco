from enum import Enum
import line_profiler
import numpy as np
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
        # print(weights)
        u = self.get_uni()
        # print(u*float(weights[-1]))
        r =  int(weights.searchsorted(u * float(weights[-1])))
        # print(r)
        return r

    def integer(self, n: int):
        return int(self.get_uni() * n)

    def rect(self, shape):
        return self.rng.random(size=shape)

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
                 maxmin: bool = False,
                 maxmin_adaptive: bool = True,
                 buffer_size: int | None = 35,
                 buffer_rule: str = "elite",
                 kernel: str = "exp",
                 adv_func: str = "reward_path",
                 importance_sampling: bool = True,
                 seed: int | None = None,
                 random_pool_size: int | None = None,
                 gradient: str = "AS"
                 ):
        self.true_distance_matrix = graph
        self.distance_matrix = graph/graph.std()
        self.n = self.distance_matrix.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.greedy_epsilon = greedy_epsilon
        self.clip_epsilon = clip_epsilon
        self.learning_rate = learning_rate
        self.pop_size = int(pop_size)
        self.grad_name = GRADNAME[gradient]

        self.epoch = 0
        self.gen_sol_score = []
        self.gen_pol_score = []

        self.rng = RandomPool(random_pool_size if random_pool_size is not None else self.n, seed)

        match kernel:
            case "identity":
                self.kernel = self.identity
            case "exp":
                self.kernel = self.expon
        self.kernel_name = kernel
        self.adv_func = ADVNAME[adv_func]

        self.phero_mat = self.rng.rect(self.distance_matrix.shape).astype(np.float64)
        self.heur_mat = (((self.distance_matrix)/self.distance_matrix.mean() + 1)**-1).astype(np.float64)
        self.prob_base = self.kernel(self.phero_mat**self.alpha * self.heur_mat**self.beta)

        self.best_sol = np.zeros(self.n)
        self.best_score = np.inf

        self.buffer_rule = buffer_rule
        self.buffer_size = int(pop_size) if buffer_size is None else int(buffer_size)
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

        self.name = "ACO"

    def expon(self, weight: NDArray):
        # print(weight.max(), weight.min(), weight.mean(), weight.std())
        return np.exp(weight - weight.max(axis=0))


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
        to_go_steps = 5
        reward_decay = 1
        valid = list(range(self.n))

        node1 = start if start is not None else self.rng.integer(self.n)
        valid.pop(valid.index(node1))
        mask[node1] = False
        trace[0] = node1

        if self.importance_sampling:
            trace_prob = np.ones(self.n, dtype=np.float64)
        else:
            trace_prob = None
        cost = 0

        for idx in range(1, self.n):
            weights = self.prob_base[node1, mask]
            # if self.epsilon < self.rng.get_uni():
            if self.greedy_epsilon > self.rng.get_uni():
                node2_idx = weights.argmax()
                # print("ep")
            else:
                node2_idx = self.rng.weighted_random(weights)
                # print("nep")
            # print(node2_idx, len(valid), valid)

            node2 = valid.pop(node2_idx)
            trace[idx] = node2
            cost += self.distance_matrix[node1, node2]

            if trace_prob is not None:
                trace_prob[idx-1] = weights[node2_idx]/(weights.sum() + 1e-4)

            match self.adv_func:
                case ADVNAME.adv_local:
                    adv[idx-1] += weights[node2_idx] - weights.mean()
                case ADVNAME.quality:
                    raise NotImplementedError
                case ADVNAME.reward:
                    adv[idx-1] += 1/self.distance_matrix[node1, node2]
                case ADVNAME.reward_to_go:
                    reward_list.append(1/self.distance_matrix[node1, node2] * reward_decay**to_go_steps)
                    if to_go_steps == len(reward_list):
                        adv[idx-to_go_steps-1] += sum(reward_list)
                        reward_list.pop(0)
                    reward_list = [(1/reward_decay) * r for r in reward_list]
                case ADVNAME.reward_baselined:
                    # print(self.distance_matrix[node1, mask])
                    adv[idx-1] += 1/self.distance_matrix[node1, node2] - (1/self.distance_matrix[node1, mask]).mean()
                    # adv[idx] += 1/self.distance_matrix[node1, node2] - (mask.sum()/self.distance_matrix[node1, mask].sum())
                case ADVNAME.reward_baselined_to_go:
                    r = 1/self.distance_matrix[node1, node2] - (1/self.distance_matrix[node1, mask]).mean()
                    # r = 1/self.distance_matrix[node1, node2] - (mask.sum()/self.distance_matrix[node1, mask].sum())
                    reward_list.append(r * reward_decay**to_go_steps)
                    if to_go_steps == len(reward_list):
                        adv[idx-to_go_steps-1] += sum(reward_list)
                        reward_list.pop(0)
                    reward_list = [(1/reward_decay) * r for r in reward_list]
                case ADVNAME.reward_decay:
                    raise NotImplementedError

            node1 = node2
            mask[node1] = False

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
                reward_list.append(1/self.distance_matrix[trace[-2], trace[-1]] * reward_decay**to_go_steps)
                reward_list.pop(0)
                reward_list = [(1/reward_decay) * r for r in reward_list]
                for i in range(to_go_steps-1):
                    adv[-i] += sum(reward_list)
                    reward_list.pop(0)
                    reward_list = [(1/reward_decay) * r for r in reward_list]
            case ADVNAME.reward_baselined_to_go:
                r = 1/self.distance_matrix[trace[-2], trace[-1]] - (1/self.distance_matrix[trace[-2], trace[-1]]).mean()
                # r = 1/self.distance_matrix[node1, node2] - (mask.sum()/self.distance_matrix[node1, mask].sum())
                reward_list.append(r * reward_decay**to_go_steps)
                reward_list.pop(0)
                reward_list = [(1/reward_decay) * r for r in reward_list]
                for i in range(to_go_steps-1):
                    adv[-i] += sum(reward_list)
                    reward_list.pop(0)
                    reward_list = [(1/reward_decay) * r for r in reward_list]
        # print("mask sum:", mask_sum)

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

                grad[node1] -= grad_node
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
        # grad = np.zeros(self.phero_mat.shape)
        # prob_base_inv = self.phero_mat**(-1)
        # for trace, l, q in zip(self.replay_buffer, self.replay_buffer_costs, self.replay_buffer_is):
        #     adv = 1/l
        #     node1 = trace[0]
        #     mask = np.ones(self.n, dtype=bool)
        #     mask[node1] = True
        #     for idx in range(1, self.n - 1):
        #         prob = self.prob_base[node1] * mask
        #         prob /= prob.sum()
        #         node2 = trace[idx]
        #         r = prob[node2]/q[idx]
        #         # clipping
        #         if (r < 1-self.clip_epsilon) and (adv < 0):
        #             continue
        #         elif (r > 1+self.clip_epsilon) and (adv > 0):
        #             continue
        #         prob[node2] -= 1
        #         x = prob_base_inv[node1]
        #         val = -1.0 * r * (adv)
        #         prob *= x * val
        #         mask[node2] = False
        #
        #         grad[node1] -= prob
        #         grad[node2] -= prob
        #         node1 = node2
        # grad /= self.buffer_size
        # self.phero_mat +=  self.rho * grad

        grad = np.zeros(self.phero_mat.shape)
        prob_base_inv = self.phero_mat**(self.alpha - 1) * self.heur_mat
        grad_node = -1 * np.ones(self.n, dtype=np.float64)
        for trace, adv, q in zip(self.replay_buffer, self.replay_buffer_adv, self.replay_buffer_is):
            # print(q)
            node1 = trace[0]
            mask = np.ones(self.n, dtype=np.float64)
            mask[node1] = False
            for idx in range(1, self.n):
                a = adv[idx-1]
                node2 = trace[idx]
                grad_node = -1 * self.prob_base[node1]
                # if sum(grad_node) is not np.nan:
                #     print(grad_node)
                grad_node /= grad_node.sum() + 1e-4
                r = (grad_node[node2] + 1e-4)/(q[idx] + 1e-4)
                # clipping
                if (r < 1-self.clip_epsilon) and (a < 0):
                    continue
                elif (r > 1+self.clip_epsilon) and (a > 0):
                    continue
                grad_node[node2] += 1
                grad_node *= a * prob_base_inv[node1] * mask

                mask[node2] = False

                grad[node1] += grad_node
                # grad[node2] += grad_node
                node1 = node2
        grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad




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
                grad[node1, node2] += 1/l
                grad[node2, node1] += 1/l
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
        self.prob_base = self.kernel(self.phero_mat**self.alpha * self.heur_mat**self.beta)
        # print("prob update", self.prob_base.min(), self.prob_base.max(), self.prob_base.std(), "|",  self.phero_mat.min(), self.phero_mat.max(), self.phero_mat.std())

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
            self.score_trace_no_heur.append([self.greedy_solution(heur=False)[1] for _ in range(10)])
            self.score_trace_heur.append([self.greedy_solution(heur=True)[1] for _ in range(10)])

        for epoch in range(max_epochs):
            self.epoch = epoch
            self.prob_base_update()
            for k in range(self.pop_size):
                self.ant_sol[k], self.cost[k], self.imp_samp[k], self.adv[k] = self.find_solution(start=start)
                c = self.get_solution_cost(self.ant_sol[k])
                if c < self.best_score:
                    self.best_score = c
                    self.best_sol = self.ant_sol[k]

            match self.adv_func:
                case ADVNAME.adv_path:
                    self.adv -= 1/self.cost.mean()
            self.gen_sol_score.append(self.best_score)
            self.gen_pol_score.append(self.greedy_solution(start=start)[1])
            self.curr_prob_mat = self.prob_base / self.prob_base.sum(axis=1)
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

            self.score_trace_no_heur.append([self.greedy_solution(heur=False)[1] for _ in range(10)])
            self.score_trace_heur.append([self.greedy_solution(heur=True)[1] for _ in range(10)])

        return self.best_sol, self.best_score

