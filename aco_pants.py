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
	print(f"total distance of tour = {total_distance}")
	return total_distance


def find_link_nodes(ordered_clusters:list) -> None:
	'''
	calculate and add link nodes to meta data of each cluster
	'''
	link_nodes = list()
	for i, c in enumerate(ordered_clusters):
		# get the next cluster in inter-cluster path
		next_c_index = None
		if i == len(ordered_clusters) - 1:
			next_c = ordered_clusters[0]
			next_c_index = 0
		else:
			next_c = ordered_clusters[i+1]
			next_c_index = i+1

		# calculate mid-point b/w centroids
		m = [(c[0][0][0]+next_c[0][0][0])/2, (c[0][0][1]+next_c[0][0][1])/2]

		# find link nodes
		c_candidates = [n for n in c[1] if n not in link_nodes]
		next_c_candidates = [n for n in next_c[1] if n not in link_nodes]
		c_link = min(c_candidates, key=lambda x:euclidean(x,m))
		next_c_link = min(next_c_candidates, key=lambda x:euclidean(x,m))

		# add link nodes to meta data of relevant cluster
		ordered_clusters[i][0].append(c_link)
		ordered_clusters[next_c_index][0].append(next_c_link)

		link_nodes.append(c_link)
		link_nodes.append(next_c_link)


def rebuild_path(ordered_clusters:list) -> None:
	''' 
	Build Hamiltonian cycle b/w clusters using linkage nodes
	I.e., convert each H.cycle within a cluster into a H.path by removing certain edges based 
	of distance of linkage nodes to next cluster's (in inter-cluster path) centroid. This 
	involves the reordering of the list of nodes in a clusters solution s.t. the link 
	node nearests to the next centroid is at the end of the list and the other is at 
	the start, whilst maintaining the order of the cluster's solution path between all 
	non-link nodes.
	'''
	for i, c in enumerate(ordered_clusters):
		# obtain centroid of next cluster after current cluster
		if i == len(ordered_clusters) - 1 :
			next_c = ordered_clusters[0]
		else:
			next_c = ordered_clusters[i+1]

		if len(c[1]) == 2:
			# re-order cluster path if necessary
			if euclidean(c[1][0], next_c[0][0]) < euclidean(c[1][1], next_c[0][0]):
				ordered_clusters[i][1] = [c[1][1], c[1][0]]	
		elif len(c[1]) > 2:
			# get cluster's link nodes
			c_links = [c[0][1], c[0][2]]
			# calculate which is closer to next cluster, assign as "end_node"
			c_links.sort(key=lambda x:euclidean(x,next_c[0][0]))
			start_node = c_links[1]
			end_node = c_links[0]

			# get locations of start & end nodes in cluster path
			s_loc = None
			e_loc = None
			
			# find index locations of start and end nodes
			for index in range(len(c[1])):
				if c[1][index] == start_node:
					s_loc = index
				elif c[1][index] == end_node:
					e_loc = index
				if s_loc is not None and e_loc is not None:
					break


			'''
			Re-order cluster path so start_node is at list[0] 
			and end_node is at list[len(list)-1]. I.e, 
			convert the Hamiltonian cycle into a H. path
			''' 
			if s_loc > e_loc:
				# everything between "right" of start node and end node,
					# as though the array is circular
				r = c[1][s_loc+1:] + c[1][0:e_loc]
				l = c[1][s_loc-1:e_loc:-1]
				ordered_clusters[i][1] = [c[1][s_loc]] + l + r + [c[1][e_loc]]
			elif e_loc > s_loc:
				r = c[1][s_loc+1:e_loc]
				if s_loc != 0:
					l = c[1][s_loc-1::-1] + c[1][len(c[1])-1:e_loc:-1]
				else:
					l = c[1][len(c[1])-1:e_loc:-1]
				ordered_clusters[i][1] = [c[1][s_loc]] + l + r + [c[1][e_loc]]


def two_opt_swap(cluster_path:list, start:int, stop:int) -> list:
	new_route = []
	for i in range(0, start):
		new_route.append(cluster_path[i])
	for i in range(stop, start, -1):
		new_route.append(cluster_path[i])
	for i in range(stop, len(cluster_path)):
		new_route.append(cluster_path[i])

	print(f"new_route: {new_route}\n")
	return new_route


def two_opt(inter_cluster_path:list, cluster_cores:list, clusters:list) -> None:
	# TODO find nodes that are part of a cluster (core nodes - see cluster's meta info)
	# Calculate midpoints between core node and its adjacent nodes 
	# For all nodes within the cluster of this core node, find one node closests to m_1 
		# and one closests to m_2
	# perform 2-opt on the clusters' nodes when node_m1 and node_m2 must stay in their
		# original order
	# remove the current code node from the ordered_clusters list 
	# and replace with the 2-opt path



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
			print(f" OLD cluster_path: {c[1]}\n")
			if c[0][0] == cluster_label:
				print("YES")
				
				start_node = sorted(c[1], key=lambda x:euclidean(x,m_prev))[len(c[1])-1]
				end_node = sorted([n for n in c[1] if n != start_node], key=lambda x:euclidean(x,m_next))[len(c[1])-2]
				cluster_path.append(start_node)
				for n in c[1]:
					if n != start_node and n != end_node:
						cluster_path.append(n)
				cluster_path.append(end_node)
				print(f" NEW cluster_path: {cluster_path}\n")
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

		print(f"NEW CLUSTER PATH 2 OPT: {cluster_path}\n")


		l = inter_cluster_path[0:c_node_loc - 1]
		mid = cluster_path
		r = inter_cluster_path[c_node_loc + 1: len(inter_cluster_path) - 1]
		inter_cluster_path = l + mid + r

	print(len(inter_cluster_path))
	# 2-opt with m1, ... ,m2 (m1,m2 in fixed pos)
	# replace core node in inter_cluster_path with 2-opt list
	






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

	print(clusterer.labels_)
	# print(clusterer.probabilities_.tolist())
	
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
	solver = pants.Solver()
	solution = solver.solve(world)

	# re-order clusters array according to inter_cluster_solution
	inter_cluster_path = [n for n in solution.tour]
	print(cluster_cores)
	
	two_opt(inter_cluster_path, cluster_cores, clusters)

	return 0

	# Find mid-point b/w clusters to link nodes b/w clusters 
	find_link_nodes(ordered_clusters)

	# For each cluster, solve aco world
	for i, c in enumerate(ordered_clusters):
		# create Hamiltonian cycle for cluster's nodes
		world = pants.World(c[1], euclidean)
		solver = pants.Solver()
		solution = solver.solve(world)
		ordered_clusters[i][1]= solution.tour

	
	rebuild_path(ordered_clusters)

	# create a list containing only the solution
	path = list()
	for c in ordered_clusters:
		for n in c[1]:
			path.append(n)

	# auxiliary functions
	draw_solution(path)  # create a graph for solution
	tour_distance(path)  # calculate total distance

if __name__ == '__main__':
	main()