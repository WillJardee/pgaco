import timeit

import numpy as np
import matplotlib.pyplot as plt

from pgaco.utils import get_graph
from pgaco.models import ACO, ACOSGD, ACOPG
from pgaco.models.__legacy import ACOSGD as ACOSGDOld
from pgaco.models.__legacy import ACO as ACOOld

def model1(graph):
    aco = ACO(graph)
    aco.run(iterations)

def model2(graph):
    aco = ACOSGD(graph)
    aco.run(iterations)

def model3(graph):
    aco = ACOPG(graph)
    aco.run(iterations)


def timefunc(func, *args):
    start = timeit.default_timer()
    func(*args)
    stop = timeit.default_timer()
    return stop-start


global iterations
iterations = 15

list1 = []
list2 = []
list3 = []
# graphsize_list = [10, 20]
graphsize_list = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                  100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for i in graphsize_list:
    graph = get_graph(i)
    print(f"running {i} on model1")
    list1.append(timefunc(model1, graph))
    print(f"running {i} on model2")
    list2.append(timefunc(model2, graph))
    print(f"running {i} on model3")
    list3.append(timefunc(model3, graph))

plt.plot(graphsize_list, list1, label="aco")
plt.plot(graphsize_list, list2, label="acosgd")
plt.plot(graphsize_list, list3, label="acopg")

plt.xlabel("nodes in graph")
plt.ylabel("runtime (s)")
plt.title("runtime with sparse")
plt.legend()
plt.tight_layout()
plt.show()
