from .runner import plot, parallel_runs
from .graph import get_graph
from .tsplib_reader import TSPGraph

def post_init_decorator(init_func):
    def wrapper(self, *args, **kwargs):
        init_func(self, *args, **kwargs)
        if hasattr(self, '_post_init'):
            self._post_init()
    return wrapper

__all__ = ["TSPGraph", "get_graph", "plot", "parallel_runs", "post_init_decorator"]

