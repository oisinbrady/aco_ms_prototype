import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import hdbscan  #  https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html#hdbscan
import seaborn as sns

from sko.ACA import ACA_TSP  # https://github.com/guofei9987/scikit-opt
import pandas as pd
from scipy import spatial


# TODO seperate progam to solve via standalone ACO strategy
# Use cProfiler to compare runtime performance

# TODO find alternative clustering algorithms

# TODO, current bottle neck within ACO of inter_cluster_nodes. 
	# I.e., Non-cluster nodes and 1 core node of each cluster.
	# "LARGE" instances incur significat wait times
	# Consider:
	# 	Iteration limit param: "limit=x"
	#	ACO library is rather old, is the implementation efficient?
	#	Reversing method of solving inter/intra-cluster paths
		# (ACO for clusters and 2-opt/3-opt(? - slower) for everything else)
	# Heuristic for pre-determining size of cluster
		# See HDBSCAN for parameter configuration
			# IDEA: grid cell mean density pre-computation to determine param(s)

# length function for weight value of edges
def euclidean(a, b):
	return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))
	

def get_nodes() -> list:
	''' 
	Construct node positional arrays, where each item is an xy coordinate 
	in a 2d euclidean space.

	'''

	# LARGE
	# http://www.math.uwaterloo.ca/tsp/world/countries.html
	# filename = "data_sets/ar9125_nodes.txt"  # Argentina: 9125_nodes.txt
	# filename = "data_sets/lu980_output.txt"  # Luxemburg: 980 "citites"
	filename = "data_sets/qa194_output.txt" # Qatar: 119 "cities"

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
		else:
			v = i + 1
			G.add_node(i+1, pos=path[i+1])

		G.add_edge(i,v)

	nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=1)

	plt.savefig("output/solution.png")
	plt.close()


def tour_distance(path:list) -> int:
	'''
	Total distance of solution
	'''
	total_distance = 0
	for i, n in enumerate(path):
		if i == len(path) - 1:
			total_distance = total_distance + euclidean(n, path[0])
		else:
			total_distance = total_distance + euclidean(n, path[i+1])
	return total_distance


def two_opt_swap(cluster_path:list, start:int, stop:int) -> list:
	'''
	Function to swap nodes during the 2-opt process
	2-opt implementation based of https://en.wikipedia.org/wiki/2-opt pseudocode 
	'''
	new_route = []
	for i in range(0, start):
		new_route.append(cluster_path[i])
	for i in range(stop, start, -1):
		new_route.append(cluster_path[i])
	for i in range(stop, len(cluster_path)):
		new_route.append(cluster_path[i])
	return new_route


def build_path(inter_cluster_path:list, cluster_cores:list, clusters:list) -> list:
	'''
	Iterates through cluster nodes. Obtains the location of core nodes within the 
	inter-cluster path (where inter_cluster_path holds one core node for each cluster).
	Obtains inter-cluster nodes adjacent to core node (not nodes within the core node's
	cluster!). Calculates the midpoints between core node and both adajcent node. 
	The start and end node for the H.path within the cluster is based of nodes that 
	are closest to these midpoints within the cluster. 2-opt is then used on the cluster.
	The start and end nodes are not included in the 2-opt search, I.e., they remain at 
	the start and end of the list representing the H.path of the cluster.
	'''

	inter_cluster_path = inter_cluster_path.tolist()
	# get cluster with core node
	for c_core in cluster_cores:
		cluster_label = c_core[1]
		core_node = c_core[0]
		c_node_loc = None

		p_node = None  # previous
		n_node = None  # next
		# get core node index in inter_cluster_path

		for i, n in enumerate(inter_cluster_path):
			if n == core_node:
				c_node_loc = i
				# get subsequent adjacent nodes
				if i == 0:
					p_node = inter_cluster_path[len(inter_cluster_path) - 1]
				else:
					p_node = inter_cluster_path[i - 1]
				if i == len(inter_cluster_path) - 1:
					n_node = inter_cluster_path[0]
				else:
					n_node = inter_cluster_path[i + 1]
				break

		# calculate m_prev, m_next: midpoints between adjacent nodes and core node
		m_prev = [(p_node[0] + core_node[0])/2, (p_node[1] + core_node[1])/2]
		m_next = [(n_node[0] + core_node[0])/2, (n_node[1] + core_node[1])/2]

		start_node = None
		end_node = None
		cluster_path = list()
		for c in clusters:
			# find cluster
			if c[0][0] == cluster_label:
				cluster_path = sorted(c[1], key=lambda x:euclidean(x,m_prev))
				break

		# perform 2-opt
		# clear caveats due to local optima, however cluster sizes should be relatively small

		improved = True
		while improved is True:
			improved = False
			best_distance = tour_distance(cluster_path)
			for i in range(1, len(cluster_path) - 1):
				for j in range(i+1, len(cluster_path) - 1):
					new_route = two_opt_swap(cluster_path, i, j)
					new_distance = tour_distance(new_route)
					if new_distance < best_distance:
						cluster_path = new_route
						best_distance = new_distance
						improved = True

		# Link the cluster's path with the inter cluster path
		l = inter_cluster_path[0:c_node_loc]
		mid = cluster_path
		r = inter_cluster_path[c_node_loc + 1: len(inter_cluster_path)]
		inter_cluster_path = l + mid + r
	
	return inter_cluster_path


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


def main():
	nodes = get_nodes()
	
	cities_np = np.array(nodes)
	clusterer = hdbscan.HDBSCAN()  # improved DBSCAN
	clusterer.fit(cities_np)  # apply clustering algorithm
	
	# configure colors to represent nodes belonging to clusters on scatter graph
	color_palette = sns.color_palette('deep', 1000)
	cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
	cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
	
	# plot scatter graph
	plt.scatter(*cities_np.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.75)
	plt.savefig("output/graph_hdbscan.png")
	
	# initialise cluster's list. 
	clusters = list()  # cluster items will contain members and meta-info
	cluster_cores = list()  # treat as "pseudo-centroids" similar to previous meanshift prototype

	for k in np.unique(clusterer.labels_):
		meta_info = [k]  # contains: label, 1 core node ("pseudo-centroid"), link nodes (later)
		cluster_nodes = []
		for i, node in enumerate(cities_np.tolist()):
			if clusterer.labels_[i] == k:
				cluster_nodes.append(node)
				if len(meta_info) == 1 and meta_info[0] != -1:
					if clusterer.probabilities_[i] == 1.0:
						# add a core node 
						meta_info.append(node)
						cluster_cores.append([node, k])
		clusters.append([meta_info, cluster_nodes])

	inter_cluster_nodes = list()
	for c in clusters:
		if c[0][0] == -1:
			for n in c[1]:
				inter_cluster_nodes.append(n)  # add all outlier nodes (not part of a cluster)
		else:
			inter_cluster_nodes.append(c[0][1])  # add core node

	
	np_icn = np.array(inter_cluster_nodes)
	distance_matrix = spatial.distance.cdist(np_icn, np_icn, metric='euclidean')
	num_points = len(np_icn)

	def cal_total_distance(routine):
		num_points, = routine.shape
		return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

	# Determine inter-cluster path via ACO
	aca = ACA_TSP(func=cal_total_distance, n_dim=len(np_icn),
	              size_pop=150, max_iter=125,
	              distance_matrix=distance_matrix)

	# running aco is clearly the bottleneck
	best_x, best_y = aca.run()

	best_points_ = np.concatenate([best_x, [best_x[0]]])

	inter_cluster_path = np_icn[best_points_, :]
	
	# build the final solution
	path = build_path(inter_cluster_path, cluster_cores, clusters)

	# TODO 2-opt for path improvement

	# auxiliary functions
	draw_solution(path)  # create a graph for solution
	print(f"Total distance: {tour_distance(path)} (abr. units)")  # calculate total distance
	print("Graph solution written to: output/solution.png")
	print("Clustering solution written to: output/graph_hdbscan.png")

if __name__ == '__main__':
	main()