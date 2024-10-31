#!/usr/bin/env python
from pathlib import Path

import numpy as np
from scipy.spatial.distance import euclidean

base_dir = Path(__file__).resolve().parent.parent
tsplib_dir = base_dir / 'tsplib'

def format_metadata(meta: str) -> dict[str, str]:
    return dict([(x[0].strip(), x[1].strip()) for x in [i.split(":") for i in meta.split("\n") if i.strip() != ""]])

def from_file(filename: str) -> tuple[np.ndarray, dict[str, str]]:
    file_path = tsplib_dir / filename
    with open(file_path, "r") as tspfile:
        point_list = []
        metadata = ""
        while "NODE_COORD_SECTION" not in (line:=tspfile.readline()): metadata += line + "\n"
        tspfile.readline()
        for line in tspfile.readlines()[:-1]:
            coord = line.strip().split()
            point_list.append([float(i) for i in coord[1::]])
    point_list = np.unique(point_list, axis=0)
    graph = np.zeros([len(point_list), len(point_list)])
    for val1, i in zip(point_list, range(len(point_list))):
        for val2, j in zip(point_list, range(len(point_list))):
            graph[i, j] = euclidean(val1, val2)
    return graph, format_metadata(metadata)

class TSPGraph:

    """Class to hold a weighted adjacency matrix from TSPLib"""

    def __init__(self, filename: str):
        """Constructor
        -------------------------
        arguments:
        filename (str): the file to read-in
        """
        self.graph, metadata = from_file(filename)
        self._name = metadata.get("NAME", None)
        self._comment = metadata.get("COMMENT", None)
        self._type = metadata.get("TYPE", None)
        self._dimensions = metadata.get("DIMENSIONS", None)
        self._edge_weight_type = metadata.get("EDGE_WEIGHT_TYPE", None)
        self._display_data_type = metadata.get("DISPLAY_DATA_TYPE", None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("tsplib_reader")
    parser.add_argument("file", help="the relative path the tsplib file to read-in.")
    parser.add_argument("save_file", help="the relative path to save the graph to.")
    args = parser.parse_args()

    tspgraph = TSPGraph(args.file)
    print(tspgraph.graph)
    print(tspgraph._name)
    print(tspgraph._comment)
    print(tspgraph._type)
    print(tspgraph._dimensions)
    print(tspgraph._edge_weight_type)


    print(tspgraph.graph.shape)
    with open(args.save_file, "w") as sfile:
        for i in tspgraph.graph:
            for j in i:
                sfile.write(f"{j:.2e},")
            sfile.write("\n")

