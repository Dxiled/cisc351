import jsonlines
import networkx as nx
import itertools

import matplotlib.pyplot as plt
import numpy as np
import collections

# =======================================================================
# PART 1: Select Data and Create Network
# Will be working with authors as nodes, co-author relationships as edges
# (tenative) Will be using all of dblp-ref-1.json
# =======================================================================

# author_list_list : a list containing all authors of each paper
# author_set       : a set containing all authors
author_list_list = []
author_set = set()

# Reads the dataset, stores the relevant information
with jsonlines.open('dblp-ref/dblp-ref-1.json') as reader:
    for obj in reader:
        try:
            author_list_list.append(obj['authors'])
            for author in obj['authors']:
                author_set.add(author)
        except:
            pass

# Creating the NetworkX Graph
G = nx.Graph()

# Adds a node for each author in author_set
# It's important to use a set to ensure no duplicates
for author in author_set:
    G.add_node(author)

# Using itertools, enumerates each pair of co-authors and adds that edge to the Graph
for author_list in author_list_list:
    relationships = itertools.combinations(author_list, 2)
    for edge in relationships:
        G.add_edge(edge[0], edge[1])

# ====================================================
# Part 2: Network analysis, reporting basic statistics
# ====================================================

print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
degree_sum = 0
for author in author_set:
    degree_sum += G.degree(author)
print("Average degree: ", degree_sum/G.number_of_nodes())
try:
    print("Radius: ", nx.algorithms.distance_measures.radius(G))
    print("Diameter: ", nx.algorithms.distance_measures.diameter(G))
except nx.exception.NetworkXError as e:
    if str(e) == "Found infinite path length because the graph is not connected":
        print("Radius and Diameter are infinite because the graph is not connected")
print("Density: ", nx.classes.function.density(G))

# Plotting degree histogram
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, axs = plt.subplots(2)

axs[0].bar(deg, cnt)
axs[1].loglog(deg, cnt)
plt.show()
# The average degree and the density show that the graph is relatively sparce, which is
# expected since it's impossible for a human to write more than a certain number of papers
# in their lifetimes (and thus it's impossible to co-author with more than a certain number
# of people). 
# The most common degree is 2, shown by the degree histogram. There are also some nodes
# which have zero edges, which indicates that some authors have not collaborated at all.
# We can see from the log-log plot that the graph follows closely to an exponential
# decay function between about 10^0.2 and 10^2, with some noise at the right end. 
# This indicates that starting from a certain point, exponentially fewer authors have 
# more co-author connections. 

# ================================
# Part 3: Node centrality analysis
# ================================

