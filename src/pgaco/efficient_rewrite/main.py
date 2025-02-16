import line_profiler
import numpy as np
from numpy.typing import NDArray
import networkx as nx
import matplotlib.pyplot as plt

class RandomPool:
    def __init__(self,
                 pool_size: int,
                 seed: None | int):
        self.rng = np.random.default_rng(seed=seed)
        self.random_pool_size = pool_size
        self.random_pool = np.zeros(pool_size)
        self.random_pool_idx = 0

    @line_profiler.profile
    def get_uni(self) -> float:
        # self.random_pool_idx = self.random_pool_idx + 1
        #
        # if self.random_pool_idx == self.random_pool_size:
        #     self.random_pool = self.rng.random(self.random_pool_size)
        #     self.random_pool_idx = 0
        #
        return self.rng.random()
        # return self.random_pool[self.random_pool_idx]

    @line_profiler.profile
    def weighted_random(self, weights: NDArray):
        weights = weights.cumsum()
        return int(weights.searchsorted(self.get_uni() * float(weights[-1])))

    @line_profiler.profile
    def integer(self, n: int):
        return int(self.get_uni() * n)

def expon(weight: NDArray):
    return np.exp(weight)

def identity(weight: NDArray):
    return weight

class Aco:
    @line_profiler.profile
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
                 buffer_size: int = 35,
                 buffer_rule: str = "elite",
                 kernel: str = "exp",
                 importance_sampling: bool = True,
                 seed: int | None = None,
                 random_pool_size: int | None = None
                 ):
        self.distance_matrix = graph
        self.n = self.distance_matrix.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.greedy_epsilon = greedy_epsilon
        self.clip_epsilon = clip_epsilon
        self.learning_rate = learning_rate
        self.pop_size = pop_size

        self.epoch = 0

        self.rng = RandomPool(random_pool_size if random_pool_size is not None else self.n, seed)

        match kernel:
            case "identity":
                self.kernel = identity
            case "exp":
                self.kernel = expon

        self.phero_mat = np.ones(self.distance_matrix.shape, dtype=float)
        self.heur_mat = ((self.distance_matrix)/self.distance_matrix.mean() + 1)**-1
        self.prob_base = self.kernel(self.phero_mat**self.alpha * self.heur_mat**self.beta)

        self.best_sol = np.zeros(self.n)
        self.best_score = np.inf

        self.buffer_rule = buffer_rule
        self.buffer_size = buffer_size
        self.replay_buffer =[]
        self.replay_buffer_costs = []
        self.replay_buffer_rating = []
        self.replay_buffer_is = []

        self.importance_sampling = importance_sampling

        self.exp_decay_grad = np.zeros(self.distance_matrix.shape)
        self.delta_grad = np.zeros(self.distance_matrix.shape)

        self.maxmin = maxmin
        self.maxmin_adaptive = maxmin_adaptive
        _, self.pheromax = self.greedy_solution()
        self.pheromin = self.pheromax / (2 * self.n)

    def greedy_solution(self, start: int | None = None):
        mask = np.ones(self.n, dtype=bool)
        trace = np.zeros(self.n + 1, dtype=int)
        valid = list(range(self.n))
        cost = 0

        node1 = start if start is not None else self.rng.integer(self.n)
        mask[node1] = False
        trace[0] = node1

        for idx in range(1, self.n):
            weights = self.prob_base[node1, mask]
            node2_idx = weights.argmax()
            node2 = valid.pop(node2_idx)
            mask[node2] = False
            trace[idx] = node2
            cost += self.distance_matrix[node1, node2]
            node1 = node2

        trace[-1] = trace[0]
        cost += self.distance_matrix[trace[-2], trace[-1]]
        return trace, cost

    @line_profiler.profile
    def find_solution(self, start = None):
        mask = np.ones(self.n, dtype=bool)
        trace = np.zeros(self.n + 1, dtype=int)
        valid = list(range(self.n))
        if self.importance_sampling:
            trace_prob = np.zeros(self.n + 1, dtype=float)
        else:
            trace_prob = None
        cost = 0

        node1 = start if start is not None else self.rng.integer(self.n)
        mask[node1] = False
        trace[0] = node1

        for idx in range(1, self.n):
            weights = self.prob_base[node1, mask]
            # if self.epsilon < self.rng.get_uni():
            if self.greedy_epsilon < self.rng.get_uni():
                node2_idx = weights.argmax()
            else:
                node2_idx = self.rng.weighted_random(weights)
            node2 = valid.pop(node2_idx)
            mask[node2] = False
            trace[idx] = node2
            cost += self.distance_matrix[node1, node2]

            if trace_prob is not None:
                trace_prob[idx - 1] = weights[node2_idx]/weights.sum()

            node1 = node2

        if trace_prob is not None:
            trace_prob[-1] = 1
        trace[-1] = trace[0]
        cost += self.distance_matrix[trace[-2], trace[-1]]

        return trace, cost, trace_prob

    def update_buffer(self):
        match self.buffer_rule:
            case "elite":
                self.push_buffer(self.cost)
            case "evict":
                self.push_buffer(-1*self.epoch * np.ones(self.pop_size))

    def push_buffer(self, ratings):
        if len(self.replay_buffer_costs) + self.pop_size > self.buffer_size:
            keep_idx = np.argpartition(np.concatenate([self.replay_buffer_rating, ratings]), self.buffer_size)[:self.buffer_size]
            self.replay_buffer = np.concatenate([self.replay_buffer, self.ant_sol])[keep_idx]
            self.replay_buffer_costs = np.concatenate([self.replay_buffer_costs, self.cost])[keep_idx]
            self.replay_buffer_rating = np.concatenate([self.replay_buffer_rating, ratings])[keep_idx]
            self.replay_buffer_probs = np.concatenate([self.replay_buffer_probs, np.repeat(self.curr_prob_mat, len(self.cost))])[keep_idx]
            if self.importance_sampling:
                self.replay_buffer_is = np.concatenate([self.replay_buffer_is, self.imp_samp])[keep_idx]
        elif len(self.replay_buffer_costs) == 0:
            self.replay_buffer = self.ant_sol
            self.replay_buffer_costs = self.cost
            self.replay_buffer_rating = ratings
            self.replay_buffer_probs = np.repeat(self.curr_prob_mat, len(self.cost))
            if self.importance_sampling:
                self.replay_buffer_is = self.imp_samp
        else:
            self.replay_buffer = np.concatenate([self.replay_buffer, self.ant_sol])
            self.replay_buffer_costs = np.concatenate([self.replay_buffer_costs, self.cost])
            self.replay_buffer_rating = np.concatenate([self.replay_buffer_rating, ratings])
            self.replay_buffer_probs = np.concatenate([self.replay_buffer_probs, np.repeat(self.curr_prob_mat, len(self.cost))])
            if self.importance_sampling:
                self.replay_buffer_is = np.concatenate([self.replay_buffer_is, self.imp_samp])


    @line_profiler.profile
    def gradient_update(self):
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
        grad /= self.buffer_size
        self.phero_mat +=  self.rho * grad

    @line_profiler.profile
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

    @line_profiler.profile
    def ppo_update(self):
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
                # clipping
                if (r < 1-self.clip_epsilon) and (adv < 0):
                    continue
                elif (r > 1-self.clip_epsilon) and (adv > 0):
                    continue
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

    @line_profiler.profile
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
        self.prob_base = self.kernel(self.phero_mat**self.alpha * self.heur_mat**self.beta)

    def maxmin_clip(self):
        self.phero_mat[np.where(self.phero_mat < self.pheromin)] = self.pheromin
        self.phero_mat[np.where(self.phero_mat > self.pheromax)] = self.pheromax

        if self.maxmin_adaptive:
            _, self.pheromax = self.greedy_solution()
            self.pheromin = self.pheromax / (2 * self.n)

    @line_profiler.profile
    def run(self, max_epochs: int = 100):
        self.ant_sol = np.zeros((self.pop_size, self.n + 1), dtype=int)
        self.cost = np.zeros(self.pop_size)
        self.imp_samp = np.zeros((self.pop_size, self.n + 1), dtype=float)
        for epoch in range(max_epochs):
            self.epoch = epoch
            self.prob_base_update()
            for k in range(self.pop_size):
                self.ant_sol[k], self.cost[k], self.imp_samp[k] = self.find_solution()
                if self.cost[k] < self.best_score:
                    self.best_score = self.cost[k]
                    self.best_sol = self.ant_sol[k]

            self.curr_prob_mat = self.phero_mat / self.phero_mat.sum(axis=1)
            self.update_buffer()

            # if self.importance_sampling:
            #     self.gradient_update_is()
            # else:
            #     self.gradient_update()
            if self.maxmin:
                self.maxmin_clip()
            # self.adaco_update()
            self.ppo_update()
            # self.tr_ppo_rb_update()

        return self.best_sol, self.best_score

def plot_cycles(G, cycle1, cycle2, pos=None):
    """
    Plot a NetworkX graph highlighting two cycles with different colors.

    Parameters:
    - G (nx.Graph): The graph to plot.
    - cycle1 (list): List of nodes in the first cycle.
    - cycle2 (list): List of nodes in the second cycle.
    - pos (dict, optional): Precomputed positions for nodes.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)  # Use a consistent layout

    plt.figure(figsize=(8, 6))

    # Draw the base graph
    nx.draw(
        G, pos, node_color="lightgray", node_size=300,
        alpha=0, with_labels=True
    )

    # Draw first cycle (e.g., red)
    cycle1_edges = [(cycle1[i], cycle1[i+1]) for i in range(len(cycle1)-1)] + [(cycle1[-1], cycle1[0])]
    nx.draw_networkx_edges(
        G, pos, edgelist=cycle1_edges,
        edge_color="black", width=2, style="dashed", label="SA"
    )

    # Draw second cycle (e.g., blue)
    cycle2_edges = [(cycle2[i], cycle2[i+1]) for i in range(len(cycle2)-1)] + [(cycle2[-1], cycle2[0])]
    nx.draw_networkx_edges(
        G, pos, edgelist=cycle2_edges,
        edge_color="grey", width=2, style="solid", label="ACO"
    )

    plt.legend()
    plt.title("Two Cycles on a Graph")
    plt.show()

if __name__ == "__main__":
    # from pgaco.utils import get_graph, plot, parallel_runs
    # graph = get_graph("berlin52.tsp")
    # G = nx.from_numpy_array(graph)

    # import datetime
    # now = datetime.datetime.now()
    # print(now.strftime("%Y-%m-%d %I:%M %p"))
    ITERS = 1000

    import tsplib95
    # problem = tsplib95.load('../tsplib/berlin52.tsp')
    graph_file = '../tsplib/att48.tsp'
    problem = tsplib95.load(graph_file)
    G = problem.get_graph()
    graph = nx.to_numpy_array(G)

    print("ITERS: ", ITERS)

    cycle = nx.approximation.simulated_annealing_tsp(G, "greedy", max_iterations=ITERS)
    cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    print("SA: ", cost)

    for i in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        aco = Aco(graph,
                  alpha = 1, # policy weight
                  beta = 2, # heur weight
                  rho = 0.9, # 1 - evap rate
                  pop_size = 10, # population size
                  greedy_epsilon = 0.1, # Greedy perc
                  clip_epsilon = 0.2, # Greedy perc
                  maxmin = True,
                  maxmin_adaptive = True,
                  buffer_size = 35,
                  buffer_rule = "elite",
                  kernel = "identity",
                  importance_sampling=True,
                  seed = None,
                  )
        aco_cycle, score = aco.run(max_epochs=ITERS)
        print("ACO search best: ", score)
        print("ACO policy: ", [aco.greedy_solution()[1] for _ in range(10)])
        print()

    # pos = {node: data['coord'] for node, data in G.nodes(data=True)}
    # plot_cycles(G, cycle[:-2], (aco_cycle+1)[:-2], pos=pos)
