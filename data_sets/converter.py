filename = "qa194.tsp"
nodes = []
with open(filename, "r") as f:
    nodes = f.readlines()[7:]
   

filtered_nodes = []
for node in nodes:
	for i, c in enumerate(node):
		if c == " ":
			node = node[i+1:]
			filtered_nodes.append(node)
			break;

f = open(f"{filename[:len(filename) - 4]}_output.txt","w+")
for n in filtered_nodes:
	n = n.strip()
	xy = n.split(' ')
	xy = [float(coordinate) for coordinate in xy]
	f.write(f"{xy[0]}, {xy[1]}\n")
