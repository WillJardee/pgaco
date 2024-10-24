from dataclasses import dataclass
import networkx as nx
import numpy as np

@dataclass
class Graph:
    graph: nx.DiGraph
    phero_init = None
    bias = None

    def __post_init__(self):
        rands = list(np.random.randint(1, 10, len(self.graph.edges)))
        for (_,_,w) in self.graph.edges(data=True):
            w['weight'] = rands.pop() 

