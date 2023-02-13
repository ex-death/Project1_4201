class Van:
    R = []
    S = []
class Person:
    pLocation = 0
    dLocation = 0
    inVan = "false"

import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_pydot import write_dot
seed=1000           # seed the graph for reproducibility, you should be doing this
G= nx.gnp_random_graph (10, .3, seed=seed )       # here we create a random binomial graph with 10 nodes and an average (expected) connectivity of 10*.3= 3.
print ( G.nodes() )

print(nx.is_connected(G))

print(G.edges())

import matplotlib.pyplot as plt

# some properties
print("node degree and node clustering")
for v in nx.nodes(G):
    print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")

print()
print("the adjacency list")
for line in nx.generate_adjlist(G):
    print(line)

links = [(u, v) for (u, v, d) in G.edges(data=True)]
pos = nx.kamada_kawai_layout(G)
nx.drawing.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue', linewidths=0.25) # draw nodes
nx.drawing.draw_networkx_edges(G, pos, edgelist=links, width=4)                                 # draw edges

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels

edge_labels = nx.get_edge_attributes(G, "weight")
print(edge_labels)
nx.draw_networkx_edge_labels(G, pos, edge_labels)

plt.show()

