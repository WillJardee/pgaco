from concurrent.futures import ProcessPoolExecutor
import numpy as np
import networkx as nx
import random
import tsplib95
from tqdm import tqdm
import matplotlib.pyplot as plt


class grad_bandit:
    def __init__(self, graph, *, learning_rate=1.0) -> None:
        self.n = graph.shape[0]
        self.true_graph = graph
        self.graph = (graph - graph.min()) / graph.std() + np.eye(self.n)
        self.heur_mat = np.zeros(graph.shape, dtype=np.float64)
        self.heur_mat -= np.eye(self.n)
        self.prob = np.exp(self.heur_mat)
        np.divide(self.prob, self.prob.sum(axis=1).reshape((-1, 1)), out=self.prob)
        self.baseline = np.zeros(graph.shape, dtype=np.float64)
        self.learning_rate = learning_rate
        self.t = 1
        self.state = 0

    def gradient_step(self, action, reward):
        self.heur_mat[self.state] -= self.learning_rate * (reward - self.baseline[self.state]) * self.prob[self.state]
        self.heur_mat[self.state, action] += self.learning_rate * (reward - self.baseline[self.state, action])
        self.baseline = (self.baseline * (self.t - 1) + reward) / self.t

    def update_prob(self):
        self.prob = np.exp(self.heur_mat)
        np.divide(self.prob, self.prob.sum(axis=1).reshape((-1, 1)), out=self.prob)

    def take_step(self, path_mask):
        prob_sum = self.prob[self.state, ~path_mask].cumsum()
        var = random.uniform(0, prob_sum[-1])
        pick = prob_sum.searchsorted(var)
        return (~path_mask).cumsum().searchsorted(pick + 1)

    def run(self, repeats, start=0):
        for _ in range(repeats):
            path_mask = np.zeros(self.n, dtype=bool)
            self.state = start
            path_mask[self.state] = 1
            for _ in range(self.n - 1):
                action = self.take_step(path_mask)
                path_mask[action] = 1
                reward = 1 / self.graph[self.state, action]
                self.gradient_step(action, reward)
                self.update_prob()
                self.state = action
                self.t += 1

    def run_policy(self, start=0):
        self.state = start
        cost = 0
        path_mask = np.ones(self.n, dtype=bool)
        trace = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            action = (self.heur_mat[self.state] * path_mask).argmax()
            trace[i] = action
            path_mask[action] = 0
            cost += self.true_graph[self.state, action]
            self.state = 0
        return cost


def run_single_instance(graph, learning_rate):
    """Runs a single instance of grad_bandit in a separate process."""
    model = grad_bandit(graph, learning_rate=learning_rate)
    model.run(100)
    return model.run_policy()


def main():
    graph_file = "../tsplib/att48.tsp"
    opt = 10628
    problem = tsplib95.load(graph_file)
    G = problem.get_graph()
    graph = nx.to_numpy_array(G)

    num_trials = 100
    learning_rate = 0.1

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_single_instance, [graph] * num_trials, [learning_rate] * num_trials), total=num_trials))

    print(np.mean(results), np.std(results))
    plt.hist(results, bins=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

