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

	plot_cluster_graph(n_clusters_, cluster_centers, cities_np, labels)

	clusters = list()  # a list of lists (list of cities in each cluster)
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

	# print(f"{ordered_clusters}\n")

	# Find mid-point b/w clusters to link nodes b/w clusters 
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


	# For each cluster, solve aco world
	for i, c in enumerate(ordered_clusters):
		# create hamiltonian cycle for cluster's nodes excluding centroid
		world = pants.World(c[1], euclidean)
		solver = pants.Solver()
		solution = solver.solve(world)
		ordered_clusters[i][1]= solution.tour

	# build hamiltonian cycle between all clusters using linkage nodes
	# essentially, each hamiltonian cycle is converted into a hamiltonian path by removing certain edges based 
		# of linkage nodes for each cluster and their distance to the next cluster centroid


	for i, c in enumerate(ordered_clusters):
		if len(c[1]) == 2:
			# obtain centroid of next cluster after current
			if i == len(ordered_clusters) - 1 :
				next_c = ordered_clusters[0]  # similar in behaviour to a circular list (TODO?)
			else:
				next_c = ordered_clusters[i+1]
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
			
			for index in range(len(c[1])):
				if c[1][index] == start_node:
					s_loc = index
				elif c[1][index] == end_node:
					e_loc = index
				if s_loc is not None and e_loc is not None:
					break

			# re-order cluster path so start_node is at list[0] and end_node is at list[len(list)-1] 
			# i.e., Hamiltonian cycle -> H. path
			if s_loc > e_loc:
				r = c[1][s_loc+1:]
				if s_loc - e_loc != 1:
					l = c[1][s_loc-1:e_loc+1:-1]
				else:
					l = []
				ordered_clusters[i][1] = [c[1][s_loc] + l + r + c[1][e_loc]]
			elif e_loc > s_loc:
				r = c[1][e_loc+1:] + c[1][0:s_loc-1]
				if e_loc - s_loc != 1:
					l = c[1][e_loc-1:s_loc+1:-1]
				else:
					l = [] 
				ordered_clusters[i][1] = [c[1][s_loc] + l + r + c[1][e_loc]]

	for c in ordered_clusters:
		print(c[1])

if __name__ == '__main__':
	main()