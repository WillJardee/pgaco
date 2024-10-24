import warnings
import networkx as nx
import numpy as np

# TODO: add seeding

# Example Generators
gauss   = lambda : np.abs(np.random.normal(loc=0, scale=1))
uniform = lambda : np.random.uniform(low=0, high=1)
const   = lambda : 1

def build_graph(n,
                start = 0,
                stop = -1,
                init_dist=None, 
                boost = "min_bump", 
                n_shortest: int = None, 
                check_shortest: bool = True):
    """Creates a weighted k-graph with n nodes initialized by some distribution 
    function. Selects a shortest path and increases the weight on each edge not 
    in that path by a constant amount, hopefully making this path the shortest 
    path.

    Args:
        n (int): Number of nodes in the graph
        init_dist (func, optional): Function that returns rules for initializing 
            edge weights. Defaults to Uniform distribution in [0,1].
        boost (int of "min_bump", optional): Value to raise all edges not in the 
            selected shortest path by. Defaults to "min_bump".
        n_shortest (int, optional): Number of elements in the shortest path. 
            Defaults to square root of n.

    Returns:
        Networkx.Graph: Networkx Graph object with fully connected graph (minus 
            self connections) with weights initialized according to init_dist 
            and all edges no in the shortest_path raised by boost
        list: List of intended shortest path.
    """
    # Define default initialization as uniform[0, 1]
    if init_dist is None: init_dist = uniform
    # Define default as sqrt
    if n_shortest is None: n_shortest = round(n**(1/2) + 0.5)
    if boost is None: boost = "min_bump"
    if stop == -1: stop = n-1

    # Generate k-graph with random weights
    dist = [(i, j, {"weight": init_dist()}) for i in range(n) for j in 
            range(n) if i > j]
    G = nx.Graph()
    G.add_edges_from(dist)

    # Get intended shortest path
    lis = np.delete(np.arange(0, n), [start, stop])
    np.random.shuffle(lis)
    shortest_path = [start] + list(lis[0:n_shortest]) + [stop]
    shortest_path_tups = [(shortest_path[i], shortest_path[i+1]) for i in 
                          range(len(shortest_path)-1)]
    print(shortest_path)

    # Derive distance from shortest path and intended shortest path if required
    target_shortest_len = sum(
            [G.edges[(shortest_path[i], shortest_path[i+1])]["weight"] for i in 
             range(len(shortest_path)-1)]
            )
        
    extra_margin = 0.0001

    if boost == "min_bump":
        if check_shortest:
            # Solve with Dijkstra - this may be computaionally unphesible if n is too large
            real_shortest = nx.shortest_path(G, shortest_path[0], shortest_path[-1], weight="weight")
            real_shortest_len = sum(
                [G.edges[(real_shortest[i], real_shortest[i+1])]["weight"] for i in 
                 range(len(real_shortest)-1)]
                 )
            boost = (target_shortest_len - real_shortest_len) * (1 + extra_margin)
        else: 
            boost = target_shortest_len * (1 + extra_margin)

    # Boost all edges not in the intended shortest path
    G_minus_shortest = [i for i in G.edges if (i not in shortest_path_tups) and 
                        (tuple(np.flip(i)) not in shortest_path_tups)]
    for i in G_minus_shortest:
        G.edges[i]["weight"] = G.edges[i]["weight"] + boost
    
    # Check that the boost was correctly done
    if check_shortest:
        if nx.shortest_path(G, shortest_path[0], shortest_path[-1], weight="weight") != shortest_path:
            warnings.warn("The desired random shortest path is not the actual shortest path. Consider changing the 'boost' parameter.")
    else:
        warnings.warn("Shortest path not checked and thus not guarenteed to be base truth.")

    print(shortest_path)
    return G, shortest_path