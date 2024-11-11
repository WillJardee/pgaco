import numpy as np

def get_graph(name: int | str, seed=42) -> np.ndarray:
    try:
        size = int(name)
        rng = np.random.default_rng(seed)
        dist_mat = rng.integers(1, 100, [size, size])
        return  dist_mat.astype(np.float64)
    except (ValueError, TypeError):
        from . import TSPGraph
        return TSPGraph(f"{name}").graph

