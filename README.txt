aco_pants.py: runs current prototype
gen_dataset.py: used to parse .tsp files for aco_pants.py

N.B., .tsp files predominantly from http://www.math.uwaterloo.ca/tsp/world/countries.html.
These files are conform to the standard tsp format following page 2 from here: 
	http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf

Rewrite may consider using https://tsplib95.readthedocs.io/en/stable/
to standardise parsing of tsp datasets. Additional libraries such as 
Networkx can be more smoothly used as a result.

