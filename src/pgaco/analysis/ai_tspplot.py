#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math
import random

from pgaco.models import ACO, ACOSGD, ACOPG, ADACO, ANTQ

def read_tsp(file_content):
    """
    Reads the TSP data from the given string and returns a list of coordinates.

    Parameters
    ----------
    file_content : str
        The content of the TSP file as a string.

    Returns
    -------
    dict
        A dictionary with node IDs as keys and coordinates (latitude, longitude) as values.
    """
    lines = file_content.strip().split("\n")
    coords = {}
    is_coord_section = False

    for line in lines:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            is_coord_section = True
            continue
        if line == "EOF":
            break
        if is_coord_section:
            parts = line.split()
            node_id = int(parts[0])
            latitude, longitude = map(float, parts[1:])
            coords[node_id] = (latitude, longitude)

    return coords

def geo_distance(coord1, coord2):
    """
    Computes the geographical distance between two coordinates using the Haversine formula.

    Parameters
    ----------
    coord1, coord2 : tuple
        Coordinates (latitude, longitude) of two points.

    Returns
    -------
    float
        The distance between the two points.
    """
    R = 6371  # Earth's radius in kilometers
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def total_distance(tour, coords):
    """
    Computes the total distance of a tour.

    Parameters
    ----------
    tour : list
        A list of node IDs representing the tour.
    coords : dict
        A dictionary of coordinates.

    Returns
    -------
    float
        Total distance of the tour.
    """
    distance = 0
    for i in range(len(tour)):
        distance += geo_distance(coords[tour[i]], coords[tour[(i + 1) % len(tour)]])
    return distance

def simulated_annealing(coords, initial_temp=1000, cooling_rate=0.995, num_iterations=10000):
    """
    Solves the TSP using simulated annealing.

    Parameters
    ----------
    coords : dict
        A dictionary of coordinates.
    initial_temp : float
        The initial temperature for simulated annealing.
    cooling_rate : float
        The rate at which the temperature cools.
    num_iterations : int
        Number of iterations for simulated annealing.

    Returns
    -------
    list
        The best tour found.
    float
        The distance of the best tour.
    """
    nodes = list(coords.keys())
    current_tour = nodes[:]
    random.shuffle(current_tour)
    current_distance = total_distance(current_tour, coords)
    best_tour = current_tour[:]
    best_distance = current_distance
    temp = initial_temp

    for _ in range(num_iterations):
        i, j = random.sample(range(len(nodes)), 2)
        new_tour = current_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_distance = total_distance(new_tour, coords)
        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temp):
            current_tour = new_tour
            current_distance = new_distance
            if current_distance < best_distance:
                best_tour = current_tour
                best_distance = current_distance
        temp *= cooling_rate

    return best_tour, best_distance

def plot_tour(coords, tour, legend):
    """
    Plots the TSP tour.

    Parameters
    ----------
    coords : dict
        A dictionary of coordinates.
    tour : list
        A list of node IDs representing the tour.
    title : str
        Title for the plot.
    """
    tour = list(tour)
    points = np.array([coords[node] for node in tour + [tour[0]]])  # Close the loop
    plt.plot(points[:, 1], points[:, 0], 'o-', label=legend)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    for node, (lat, lon) in coords.items():
        plt.text(lon, lat, f"{node}", fontsize=9)


TSP_FILE = "tsplib/burma14.tsp"
# TSP_FILE = "tsplib/att48.tsp"
tsp_data = ""

with open(TSP_FILE, "r") as readfile:
    for line in readfile.readlines():
        tsp_data += line.strip() + "\n"

print(tsp_data)

# tsp_data = """
# NAME: burma14
# TYPE: TSP
# COMMENT: 14-Staedte in Burma (Zaw Win)
# DIMENSION: 14
# EDGE_WEIGHT_TYPE: GEO
# EDGE_WEIGHT_FORMAT: FUNCTION
# DISPLAY_DATA_TYPE: COORD_DISPLAY
# NODE_COORD_SECTION
#    1  16.47       96.10
#    2  16.47       94.44
#    3  20.09       92.54
#    4  22.39       93.37
#    5  25.23       97.24
#    6  22.00       96.05
#    7  20.47       97.02
#    8  17.20       96.29
#    9  16.30       97.38
#   10  14.05       98.12
#   11  16.53       97.38
#   12  21.52       95.59
#   13  19.41       97.13
#   14  20.09       94.55
# EOF
# """

SEED = 42
MAX_ITER = 100

# Main execution
coords = read_tsp(tsp_data)
print(coords)
# plot_tour(coords, list(coords.keys()), "TSP Problem: Initial Points")

# Solve TSP using simulated annealing
best_tour, best_distance = simulated_annealing(coords)

num_points = len(coords)
distance_matrix = np.zeros((num_points, num_points))

print(f"Best Tour (SA): {best_tour}")
print(f"Best Distance (SA): {best_distance:.2f} km")
plot_tour(coords, best_tour, "simulated annealing")

for i in range(num_points):
    for j in range(num_points):
        coord1 = coords[i + 1]
        coord2 = coords[j + 1]
        distance_matrix[i, j] = geo_distance(coord1, coord2)


aco = ACOPG(distance_matrix,
            minmax=True,
            slim = False,
            seed = SEED)


best_distance, best_tour  = aco.run(MAX_ITER)

print(best_tour)
print(best_tour + 1)

print(f"Best Tour (ACOPG): {best_tour + 1}")
print(f"Best Distance (ACOPG): {best_distance:.2f} km")

# Plot the solution
plot_tour(coords, best_tour + 1, "ACOPG")


plt.title("TSP tours")
plt.legend()
plt.grid()
plt.show()



