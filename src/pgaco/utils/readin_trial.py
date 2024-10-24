#TODO: change to pickle

class edge:
    def __init__(self, weight, pheromone, bias) -> None:
        self.weight = weight
        self.bias = bias
        self.pheromone = []
        self.pheromone.append(pheromone)

def read_in(file_name):
    file = open(file_name, "r")
    shortest_path = file.readline()
    if 'Unknown' in shortest_path:
        shortest_path = None
    else:
        shortest_path = shortest_path.split("[")[-1].strip("]\n")
        shortest_path = shortest_path.split(", ")
        shortest_path = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
    
    file.readline() # read "Initial state is" line
    file.readline() # read "State after iteration 0"

    graph = {}
    while True:
        line = file.readline()
        if line == "\n":
            file.readline()
            break
        line = line.strip().strip("(").strip(")").split(", ")
        weight = [float(i.strip("{").split(": ")[-1]) for i in line if "weight" in i][0]
        phero = [float(i.split(": ")[-1]) for i in line if "pheromone" in i][0]
        bias = [float(i.strip("}").split(": ")[-1]) for i in line if "bias" in i][0]
        graph[(line[0], line[1])] = edge(weight=weight, pheromone=phero, bias=bias)

    for line in file.readlines():
        if line == "\n" or "State after iteration" in line:
            continue
        line = line.strip().strip("(").strip(")").split(", ")
        try:
            e = graph[(line[0], line[1])]
        except:
            e = graph[(line[1], line[0])]
        e.pheromone.append([float(i.split(": ")[-1]) for i in line if "pheromone" in i][0])

    return graph, shortest_path