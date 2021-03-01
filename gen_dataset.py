import random

# using file as template
filename = "city_xy"
nodes = []

with open(filename, 'r') as f:
	for city in f:
		city = city.strip()
		xy = city.split(',')
		xy = [float(coordinate) for coordinate in xy]
		nodes.append(xy)
	f.close()


f = open("data_sets/gen_cities.tsp","w+")
for n in nodes:
	f.write(f"{n[0]}, {n[1]}\n")

NEW_NODES = 100
for i in range(NEW_NODES):
	x = random.uniform(0, -80)
	y = random.uniform(-30, 50)
	f.write(f"{x}, {y}\n")
f.close() 
