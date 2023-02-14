import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


### Tutorial Block
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_pydot import write_dot
seed=1000           # seed the graph for reproducibility, you should be doing this
numberOfNodes = 10
G= nx.gnp_random_graph (numberOfNodes, .3, seed=seed )       # here we create a random binomial graph with 10 nodes and an average (expected) connectivity of 10*.3= 3.
print ( G.nodes() )

print(nx.is_connected(G))

print(G.edges())



# some properties
print("node degree and node clustering")
for v in nx.nodes(G):
    print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")

print()
print("the adjacency list")
for line in nx.generate_adjlist(G):
    print(line)

links = [(u, v) for (u, v, d) in G.edges(data=True)]
pos = nx.kamada_kawai_layout(G, dim= 2)
nx.drawing.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue', linewidths=0.25) # draw nodes
nx.drawing.draw_networkx_edges(G, pos, edgelist=links, width=4)                                 # draw edges

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels

edge_labels = nx.get_edge_attributes(G, "weight")
print(edge_labels)
nx.draw_networkx_edge_labels(G, pos, edge_labels)
### End Tutorial Block

### Start Algorithms
import random

### Class Declarations
class Van:
    def __init__(self, startPos):
        self.currentNode = startPos
        self.Hist = [startPos]

    nextNode = 0
    xyData = []
    R = []
    S = []
class Person:
    def __init__(self, pickUp, dropOff):
        pLocation = pickUp
        dLocation = dropOff
    inVan = "false"


# interpolate coordinates between two point with time as percentage
# ex. time = .5 = halfway between
def anim_lerp(a, b, time, pos):
    a = (pos[a][0],pos[a][1])
    b = (pos[b][0],pos[b][1])
    newab = tuple(np.subtract(b,a))
    newab = tuple(i * time for i in newab)
    newab = tuple(np.add(a, newab))
    return newab

#initialize vans here
numberOfVans = 1 # set to 30 later
van = []
for i in range(numberOfVans):
    van.append(Van(random.randrange(numberOfNodes)))

# van[0].Hist.pop()
# van[0].Hist.append(2)
# van[0].Hist.append(8)
# van[0].Hist.append(1)
# van[0].Hist.append(4)
# van[0].Hist.append(9)

runTime = 100 #number of ticks
#ticks script here - ticks * 4 to accomodate for animation
for tick in range(runTime * 4):
    if(tick % 4 == 0): # every 4 "ticks" = 1 clock tick
        #insert scripts for every tick here
        print("Tick: " + str(tick / 4))
    if(tick % 16 == 0):
        #insert scripts for every 4 clock ticks here
        print("4thTick: " + str(tick / 16))
    #animation scripts here
    # records each location of van for each tick
    for i in range(numberOfVans):
        van[i].xyData.append(anim_lerp(van[i].currentNode, van[i].nextNode, ((tick % 8)/4), pos))


### ANIMATION
fig = plt.gcf()
ax = fig.gca()

# vanXY = np.zeros((runTime * 4, numberOfVans))
graphVans = {}
for n in range(numberOfVans):
    graphVans[n] = plt.Circle((van[n].xyData[0]), .05)


def anim_init():
    for n in range(numberOfVans):
        graphVans[n].center = van[n].xyData[0]
        ax.add_patch(graphVans[n])
    return graphVans.values()


def anim_run(i):
    for n in range(numberOfVans):
        graphVans[n].center = van[n].xyData[i]
    return graphVans.values()


# circle = plt.Circle((anim_lerp(0, 3, .5, pos)), .05, zorder=10)
# circle.center = anim_lerp(0, 3, 1, pos)
# ax.add_patch(circle)

anim = FuncAnimation(fig, func=anim_run, init_func=anim_init, frames=runTime * 4, interval=10)

plt.show()

