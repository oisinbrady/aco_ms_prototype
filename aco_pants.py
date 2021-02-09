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

	print("number of estimated clusters : %d" % n_clusters_)

	plot_cluster_graph(n_clusters_, cluster_centers, cities_np, labels)
	print(labels)

	print(cluster_centers)

	clusters = dict()  # a dict of lists (list of cities in each cluster)
	# N.B., the first value for each cluster key will be the cluster centroid 
	for k in range(n_clusters_):
		cities = list()
		cities.append(cluster_centers[k].tolist())  # add centroid value
		for i, city_xy in enumerate(cities_np.tolist()):
			if labels[i] == k:
				cities.append(city_xy)  # add city belonging to current cluster
		clusters[k] = cities


	print("cluster output here!!")
	print(clusters)

	# Determine path between clusters via ACO - using centroids as artificial nodes
	world = pants.World(cluster_centers.tolist(), euclidean)
	solver = pants.Solver()
	solution = solver.solve(world)

	# solution contains: total distance(.distance), nodes visited order (.tour), edges visited order(.path)
	print(f"{solution.tour}")
	
	# todo: follow path and determine 'real' nodes to link b/w clusters 
	linkage_nodes = list()  # all nodes that link clusters
	m_points = list()

	for i, c1 in enumerate(solution.tour):
		# find the mid-point between both clusters
		if i + 1 == len(solution.tour):
			c2 = solution.tour[0]  # last cluster in path back to start cluster
		else:
			c2 = solution.tour[i + 1]  # current cluster to next in path
		m = [(c1[0]+c2[0])/2,(c1[1]+c2[1])/2]
		# print(f"m-point b/w c{i} and c{i+1}: {m}")
		m_points.append(m)


		# (m-point, consider nodes w/n range b/w centroid and m-point [half of cluster's nodes])
		# store these nodes (in order) as 'linkage nodes'

	# todo: for each cluster, solve aco world
		# N.B. edge case needed for clusters of size 2 and 1 
		# if |cluster| = 2, simply link nodes, if 1, link to next cluster in path

	# todo: build full path, using linked nodes
		# add edges between 'linkage' nodes
		# remove 'extra' path ways for all nodes with edges > 2
		# remove the longer edge, or remove if edge connects to another 'linkage' node
		# there should now exist exactly two nodes that have 1 edge; link these for a complete path


if __name__ == '__main__':
	main()