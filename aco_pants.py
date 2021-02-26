import pants
import math
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import networkx as nx
import hdbscan
import seaborn as sns


# TODO seperate progam to solve via standalone ACO strategy
# Use cProfiler to compare runtime performance

# TODO find alternative clustering algorithms
# 	IDEA: DB clustering, will need to calculate centroids if using current solution


# length function for weight value of edges
def euclidean(a, b):
	return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))

# create the cities' coordinates array
	# "P01" https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html

def get_cities() -> list:
	# 9125 "cities" in Argentina, FAILS b/c only one cluster produced
	# Therefore, need to reconsider clustering algorithm
	# filename = "data_sets/ar9125_nodes.txt"  

	filename = "data_sets/qa194_output.txt" # Qatar: 119 "cities"
	# filename = "gen_cities.txt"  # 116 cities
	# filename = "city_xy_2"  # 48 cities
	# filename = "city_xy"  # 15 cities
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

	plt.savefig("solution.png")


def tour_distance(path:list) -> int:
	total_distance = 0
	for i, n in enumerate(path):
		if i == len(path) - 1:
			total_distance = total_distance + euclidean(n, path[0])
		else:
			total_distance = total_distance + euclidean(n, path[i+1])
	return total_distance


def two_opt_swap(cluster_path:list, start:int, stop:int) -> list:
	new_route = []
	for i in range(0, start):
		new_route.append(cluster_path[i])
	for i in range(stop, start, -1):
		new_route.append(cluster_path[i])
	for i in range(stop, len(cluster_path)):
		new_route.append(cluster_path[i])
	return new_route


def two_opt(inter_cluster_path:list, cluster_cores:list, clusters:list) -> list:
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
			# print(f" OLD cluster_path: {c[1]}\n")
			if c[0][0] == cluster_label:
				# print("YES")
				
				start_node = sorted(c[1], key=lambda x:euclidean(x,m_prev))[0]
				end_node = sorted([n for n in c[1] if n != start_node], key=lambda x:euclidean(x,m_next))[0]
				cluster_path.append(start_node)
				for n in c[1]:
					if n != start_node and n != end_node:
						cluster_path.append(n)
				cluster_path.append(end_node)
				# print(f" NEW cluster_path: {cluster_path}\n")
				break

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

		# print(f"NEW CLUSTER PATH 2 OPT: {cluster_path}\n")


		l = inter_cluster_path[0:c_node_loc]
		mid = cluster_path
		r = inter_cluster_path[c_node_loc + 1: len(inter_cluster_path)]
		inter_cluster_path = l + mid + r
	
	return inter_cluster_path


def main():
	nodes = get_cities()

	# apply clustering algorithm
	cities_np = np.array(nodes)
	clusterer = hdbscan.HDBSCAN()
	clusterer.fit(cities_np)
	color_palette = sns.color_palette('deep', 1000)
	cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
	cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
	# plot as scatter graph
	plt.scatter(*cities_np.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.75)
	plt.savefig("graph_hdbscan.png")
	
	# initialise cluster's list. Holds paths and meta data. I.e. link nodes 
	clusters = list()
	cluster_cores = list()  # pseudo-centroids and the cluster they belong to

	for k in np.unique(clusterer.labels_):
		meta_info = [k]  # contains: label, 1 core node (pseudo-centroid), link nodes (later)
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

	# Determine inter-cluster path via ACO
	world = pants.World(inter_cluster_nodes, euclidean)
	solver = pants.Solver(limit=3000)  # limit = 100
	solution = solver.solve(world)

	# re-order clusters array according to inter_cluster_solution
	inter_cluster_path = [n for n in solution.tour]
	
	path = two_opt(inter_cluster_path, cluster_cores, clusters)

	# print(path)

	# auxiliary functions
	draw_solution(path)  # create a graph for solution
	print(f"{tour_distance(path)}")  # calculate total distance

if __name__ == '__main__':
	main()