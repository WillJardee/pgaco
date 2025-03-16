#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import networkx as nx

from pgaco.utils import  get_graph

graph = "att48.tsp"
distance_matrix = get_graph(graph)

G = nx.from_numpy_array(distance_matrix)
cycle = nx.approximation.simulated_annealing_tsp(G, "greedy")
cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))




