import pants
import math
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import networkx as nx


def plot_cluster_graph(n_clusters_, cluster_centers, cities_np, labels) -> None:
	plt.figure(1)
	plt.clf()

	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
	    my_members = labels == k
	    cluster_center = cluster_centers[k]
	    plt.plot(cities_np[my_members, 0], cities_np[my_members, 1], col + '.')
	    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=14)
	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.savefig("graph.png")
	plt.close()


# length function for weight value of edges
def euclidean(a, b):
	return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))

# create the cities' coordinates array
	# "P01" https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html

def get_cities() -> list:
	# filename = "gen_cities.txt"  # 116 cities
	# filename = "city_xy_2"  # 48 cities
	filename = "city_xy"  # 15 cities
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


def tour_distance(path:list) -> None:
	total_distance = 0
	for i, n in enumerate(path):
		if i == len(path) - 1:
			total_distance = total_distance + euclidean(n, path[0])
		else:
			total_distance = total_distance + euclidean(n, path[i+1])
	print(f"total distance of tour = {total_distance}")


def find_link_nodes(link_nodes:list, ordered_clusters:list):
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


def rebuild_path(ordered_clusters:list):
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


def main():
	nodes = get_cities()

	# apply clustering algorithm

	cities_np = np.array(nodes)
	ms = MeanShift(bin_seeding=True)
	ms.fit(cities_np)  # use mean shift algorithm
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_  # centroids of all clusters

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	plot_cluster_graph(n_clusters_, cluster_centers, cities_np, labels)

	# initialise cluster's list. Holds paths and meta data. I.e. link nodes 
	clusters = list()
	for k in range(n_clusters_):
		meta_info = list()  # to contain centroid xy, and link nodes (2) xy
		meta_info.append(cluster_centers[k].tolist())  # add centroid value
		cities = list()  # xy coordinates of all city nodes
		for i, city_xy in enumerate(cities_np.tolist()):
			if labels[i] == k:
				cities.append(city_xy)  # add city belonging to current cluster
		clusters.append([meta_info, cities])


	# Determine inter-cluster path via ACO - using centroids as artificial nodes
	world = pants.World(cluster_centers.tolist(), euclidean)
	solver = pants.Solver()
	inter_cluster_solution = solver.solve(world)

	# re-order clusters array according to inter_cluster_solution
	ordered_clusters = list()
	for c in inter_cluster_solution.tour:
		for k in clusters:
			if c == k[0][0]:
				ordered_clusters.append(k)

	# Find mid-point b/w clusters to link nodes b/w clusters 
	link_nodes = list()
	find_link_nodes(link_nodes, ordered_clusters)

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