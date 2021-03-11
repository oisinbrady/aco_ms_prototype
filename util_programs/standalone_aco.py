import math  # for distance measurements
import numpy as np  # for networkx, matplotlib, and ACO alg.
import matplotlib.pyplot as plt  # graph plotting
import networkx as nx  # solution graph
import hdbscan  #  https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html#hdbscan
import seaborn as sns  # for scatter graph cluster colors

# ACO
from sko.ACA import ACA_TSP  # https://github.com/guofei9987/scikit-opt
from scipy import spatial

def get_nodes() -> list:
	''' 
	Construct node positional arrays, where each item is an xy coordinate 
	in a 2d euclidean space.

	'''

	# LARGE
	# http://www.math.uwaterloo.ca/tsp/world/countries.html
	# filename = "data_sets/ar9125_nodes.txt"  # Argentina: 9125_nodes.txt
	filename = "data_sets/lu980_output.txt"  # Luxemburg: 980 "citites"
	# filename = "data_sets/qa194_output.txt" # Qatar: 119 "cities"

	# CUSTOM - randomized cities built ontop of "P01" template
	# filename = "data_sets/gen_cities.txt"  # 116 cities

	# SMALL
	# https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
	# filename = "data_sets/city_xy_2"  # "ATT48" American US state capitals
	# filename = "data_sets/city_xy"  # # "P01": 15 cities
	nodes = []

	with open(filename, 'r') as f:
		for city in f:
			city = city.strip()
			xy = city.split(',')
			xy = [float(coordinate) for coordinate in xy]
			nodes.append(xy)

	return nodes


def draw_solution(path:list) -> None:
	G = nx.Graph()
	for i, n in enumerate(path):
		G.add_node(i, pos=n)
		if i == len(path) - 1:
			v = 0
			G.add_node(0, pos=path[0])
			# from last to first node
		else:
			v = i + 1
			G.add_node(i+1, pos=path[i+1])
			# from current node to next node in path

		G.add_edge(i,v)

	nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=False, node_size=1)

	plt.savefig("output/solution.png")
	plt.close()


nodes = get_nodes()
cities_np = np.array(nodes)

distance_matrix = spatial.distance.cdist(cities_np, cities_np, metric='euclidean')

def cal_total_distance(routine):
		num_points, = routine.shape
		return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

aca = ACA_TSP(func=cal_total_distance, n_dim=len(nodes),
	              size_pop=150, max_iter=125,
	              distance_matrix=distance_matrix)

# bottleneck No.1
print("running ACO...")
best_x, best_y = aca.run()
best_points_ = np.concatenate([best_x, [best_x[0]]])
path = cities_np[best_points_, :]

draw_solution(path)  # create a graph for solution