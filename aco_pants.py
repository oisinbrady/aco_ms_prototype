import pants
import math
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


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


# length function for weight value of edges
def euclidean(a, b):
	return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))


# create the cities' coordinates array
	# "P01" https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html

def get_cities() -> list:
	filename = "city_xy"
	nodes = []

	with open(filename, 'r') as f:
		for city in f:
			city = city.strip()
			xy = city.split(',')
			xy = [float(coordinate) for coordinate in xy]
			nodes.append(xy)

	return nodes

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

	# print("number of estimated clusters : %d" % n_clusters_)

	plot_cluster_graph(n_clusters_, cluster_centers, cities_np, labels)
	# print(labels)

	# print(cluster_centers)

	clusters = dict()  # a dict of lists (list of cities in each cluster)
	# N.B., the first value for each cluster key will be the cluster centroid 
	for k in range(n_clusters_):
		cities = list()
		cities.append(cluster_centers[k].tolist())  # add centroid value
		for i, city_xy in enumerate(cities_np.tolist()):
			if labels[i] == k:
				cities.append(city_xy)  # add city belonging to current cluster
		clusters[k] = cities


	# print("cluster output here!!")
	# print(clusters)

	# Determine path between clusters via ACO - using centroids as artificial nodes
	world = pants.World(cluster_centers.tolist(), euclidean)
	solver = pants.Solver()
	solution = solver.solve(world)

	# solution contains: total distance(.distance), nodes visited order (.tour), edges visited order(.path)
	# print(f"\n{solution.tour}")
	# print(f"\n{solution.path}")
	# for e in solution.path:
	# 	print(vars(e))

	# re-order clusters array according to tour solution of centroids
	ordered_clusters = dict()
	for c in solution.tour:
		for k in clusters:
			if c == clusters[k][0]:
				ordered_clusters[k] = clusters[k]

	# print(f"{ordered_clusters}\n")

	# Find the mid-point between clusters & determine nodes to link b/w clusters 
	linkage_nodes = list()  # all nodes that link clusters
	temp = list(ordered_clusters.items())
	for i in range(len(ordered_clusters)):
		c1 = temp[i]
		if i + 1 == len(ordered_clusters):
			c2 = temp[0]  # last cluster in path back to start cluster, similar to a circular array
		else:
			c2 = temp[i + 1]  # current cluster to next in path

		m = [(c1[1][0][0]+c2[1][0][0])/2, (c1[1][0][1]+c2[1][0][1])/2]  # mid point b/w centroids
		# print(f"m-point = {m}")

		# For clusters c1, c2, find node with shortest distance to m-point
		c1_non_link = list(filter(lambda x : x not in linkage_nodes, c1[1][1:]))  # c1_nodes ∉ linkage_nodes
		c1_min = min(c1_non_link, key=lambda x:euclidean(x,m))

		c2_non_link = list(filter(lambda x : x not in linkage_nodes, c2[1][1:]))  # c2_nodes ∉ linkage_nodes
		c2_min = min(c2_non_link, key=lambda x:euclidean(x,m))

		# get the cluster each link belongs to
		label_c1 = c1[0]
		label_c2 = c2[0]
		linkage_nodes.append((label_c1, c1_min))
		linkage_nodes.append((label_c2, c2_min))

	# For each cluster, solve aco world
	cluster_paths = list()
	for c in list(ordered_clusters.items()):
		# create hamiltonian cycle for cluster's nodes excluding centroid
		world = pants.World(c[1][1:], euclidean)
		solver = pants.Solver()
		solution = solver.solve(world)
		cluster_paths.append(solution.tour)

	print(f"{ordered_clusters}\n")  # hamiltonian cycle of CENTROIDS
	print(f"{cluster_paths}\n")  # hamiltonian cycle of each CLUSTER
	print(linkage_nodes)  # nodes that link between clusters (contains cluster labels)
	
	# N.B. edge case needed for clusters of size 2 and 1 
	# if |cluster| = 2, simply link nodes, if 1, link to next cluster in path


	# build hamiltonian cycle between all clusters using linkage nodes
	# essentially, each hamiltonian cycle is converted into a hamiltonian path by removing certain edges based 
		# of linkage nodes for each cluster and their distance to the next cluster centroid
	# TODO k is cluster number and so is not in order, use i instead of subscripting cluster_paths
	i = 0  # index of cluster in cluster_paths
	for cluster_id in ordered_clusters:
		# when |ordered_clusters[cluster_id]| = 2
		if len(cluster_paths[i]) == 2:
			if i == len(cluster_paths):
				next_kc = ordered_clusters[0][0]  # next cluster centroid will be at start of list
			else:
				next_kc = ordered_clusters[i+1][0]  # centroid of next cluster
			# re-order cluster path if necessary
			if euclidean(cluster_paths[i][0], next_kc) < euclidean(cluster_paths[i][1], next_kc):
				cluster_paths[i] = [cluster_paths[i][1], cluster_paths[i][0]]	
		elif len(cluster_paths[i]) > 2:
			# find current cluster's linkage nodes: l1 and l2
			cluster_link_nodes = list()		
			for n in linkage_nodes:
				if linkage_nodes[n][0] == cluster_id:  # if linkage node has current cluster_id label
					cluster_link_nodes.append(linkage_nodes[n][1])
				if len(cluster_link_nodes) == 2:
					break  # as all clusters have exactly 2 link nodes
			
			# calculate which one is closer to next cluster, assign as "end_node"
			c = ordered_clusters[cluster_id][0]  # centroid of next cluster
			cluster_link_nodes.sort(lambda x:euclidean(x,c))
			start_node = cluster_link_nodes[0]
			end_node = cluster_link_nodes[1]
			# get locations of start & end nodes in cluster path
			s_loc = None
			e_loc = None
			for index in range(len(cluster_paths[i])):
				if cluster_paths[i][n] == start_node:
					s_loc = n
				elif cluster_paths[i][n] == end_node:
					e_loc = n
				if s_loc is not None and e_loc is not None:
					break

			l1 = None
			if s_loc > e_loc:
				l1 = cluster_paths[i][s_loc+1:] + cluster_paths[i][0:e_loc-1]
			# re-order cluster path so start_node is at list[0] and end_node is at list[len(list)-1] (Hamiltonian cycle -> H. path)



		i = i + 1

				


	# TODO: build full path, using linked nodes - READ NOTES FILE
		# add edges between 'linkage' nodes
		# remove 'extra' path ways for all nodes with edges > 2
		# remove the longer edge, or remove if edge connects to another 'linkage' node
		# there should now exist exactly two nodes that have 1 edge; link these for a complete path


if __name__ == '__main__':
	main()