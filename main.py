import numpy as np
import matplotlib.pyplot as plt
import random
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


### Class Declarations
class Van:
    def __init__(self, startPos):
        self.currentNode = startPos

    nextNode = 0
    xyData = []
    R = []
    S = []
    mid = False # distance traveled from current node to next node, true meaning distance = halfway
class Person:
    def __init__(self, pickUp, dropOff):
        self.pLocation = pickUp
        self.dLocation = dropOff
        self.inVan = False


# interpolate coordinates between two point with time as percentage
# ex. time = .5 = halfway between
def anim_lerp(a, b, time, pos):
    a = (pos[a][0],pos[a][1])
    b = (pos[b][0],pos[b][1])
    newab = tuple(np.subtract(b, a))
    newab = tuple(j * time for j in newab)
    newab = tuple(np.add(a, newab))
    return newab

def generatePeople(people):
    # need 7.5 people needing rides per tick
    if (random.randint(0, 1) == 1):
        # make 8 people
        for i in range(2): # make 8 later
            # generate random pickup & dropoff
            pickup = random.randint(0, 9)
            dropoff = random.randint(0, 9)
            # make sure they are different
            while pickup == dropoff:
                dropoff = random.randint(0, 9)
            # create person and add to waitlist
            guy = Person(pickup, dropoff)
            people.append(guy)
    else:
        # make 7 people
        for i in range(2): #make 7 later
            # generate random pickup & dropoff
            pickup = random.randint(0, 9)
            dropoff = random.randint(0, 9)
            # make sure they are different
            while pickup == dropoff:
                dropoff = random.randint(0, 9)
            # create person and add to waitlist
            guy = Person(pickup, dropoff)
            people.append(guy)

def assignVan(people, van):
    cost = 9999 # set cost high
    while people: # while people are still on waitlist
        x = people.pop() # remove person from waitlist
        for y in van: # run through vans
            tempCost = nx.astar_path_length(G, y.currentNode, x.pLocation) # find a* path cost
            if tempCost < cost: # if the cost of the new path is less than the current cost, set this as new shortest path
                tempVan = y
        tempVan.R.append(x) # add person to R list
   # for x in van: # print R list
        # print("R:")
      #  for y in x.R:
      #      print(y.pLocation)

def pickupSchedule(van):
    if len(van.R) == 1 and len(van.S) <=2: # if only p1 in R queue
        print("Got Here 1")
        van.S.append(van.R.pop(0)) # append p1 pLocation to S
    if len(van.R) == 2 and len(van.S) <=1: # if p1 and p2 in R queue
        print("Got Here 2")
        t1 = van.R.pop(0) # t(#) for temp person pickup location, pops queue
        t2 = van.R.pop(0)
        p1 = nx.astar_path_length(G, van.currentNode, t1.pLocation) # p(#) for the distance from van to corresponding pLocation
        p2 = nx.astar_path_length(G, van.currentNode, t2.pLocation)
        if p1 < p2: # if distance to p1 is less than distance to p2
            van.S.append(t1) # append p1 then p2 pLocation to S
            van.S.append(t2)
        else:
            van.S.append(t2)
            van.S.append(t1)
    if len(van.R) >= 3 and len(van.S) ==0 : # if p1, p2 and p3 in R queue
        print("Got Here 3")
        t1 = van.R.pop(0)
        t2 = van.R.pop(0)
        t3 = van.R.pop(0)
        p1 = nx.astar_path_length(G, van.currentNode, t1.pLocation)
        p2 = nx.astar_path_length(G, van.currentNode, t2.pLocation)
        p3 = nx.astar_path_length(G, van.currentNode, t3.pLocation)
        if p1 < p2 and p1 < p3:
            van.S.append(t1)

            tp2 = nx.astar_path_length(G, t1.pLocation, t2.pLocation)
            tp3 = nx.astar_path_length(G, t1.pLocation, t3.pLocation)

            if tp2 < tp3:
                van.S.append(t2)
                van.S.append(t3)
            else:
                van.S.append(t3)
                van.S.append(t2)
        elif p2 < p1 and p2 < p3:
            van.S.append(t2)

            tp1 = nx.astar_path_length(G, t2.pLocation, t1.pLocation)
            tp3 = nx.astar_path_length(G, t2.pLocation, t3.pLocation)

            if tp1 < tp3:
                van.S.append(t1)
                van.S.append(t3)
            else:
                van.S.append(t3)
                van.S.append(t1)
        else:
            van.S.append(t3)

            tp1 = nx.astar_path_length(G, t3.pLocation, t1.pLocation)
            tp2 = nx.astar_path_length(G, t3.pLocation, t2.pLocation)

            if tp1 < tp2:
                van.S.append(t1)
                van.S.append(t2)
            else:
                van.S.append(t2)
                van.S.append(t1)


def dropScheduler(van):
    size = len(van.S)
    if size < 1:
        return 0
    t1 = van.S.pop(0)
    p1 = nx.astar_path_length(G, van.currentNode, t1.dLocation)
    if size > 1:
        t2 = van.S.pop(0)
        if t2.inVan == False:
            p2 = nx.astar_path_length(G, van.currentNode, t2.pLocation)
            p1b = nx.astar_path_length(G, t2.pLocation, t1.dLocation)
        else:
            p2 = nx.astar_path_length(G, van.currentNode, t2.dLocation)
            p1b = nx.astar_path_length(G, t2.dLocation, t1.dLocation)

    if size == 3:
        t3 = van.S.pop(0)
        if t2.inVan == False:
            if t3.inVan == False:
                p3b = nx.astar_path_length(G, t2.pLocation, t3.pLocation)
            else:
                p3b = nx.astar_path_length(G, t2.pLocation, t3.dLocation)
        else:
            if t3.inVan == False:
                p3b = nx.astar_path_length(G, t2.dLocation, t3.pLocation)
            else:
                p3b = nx.astar_path_length(G, t2.dLocation, t3.dLocation)


    if size == 1:
        van.S.append(t1)
    elif size > 1:
        if size == 3:
            if p1 < p2:
                van.S.append(t1)
                van.S.append(t2)
                van.S.append(t3)
            elif p1b < p3b:
                van.S.append(t2)
                van.S.append(t1)
                van.S.append(t3)
            else:
                van.S.append(t2)
                van.S.append(t3)
                van.S.append(t1)
        else:
            if p1 < p2:
                van.S.append(t1)
                van.S.append(t2)
            else:
                van.S.append(t2)
                van.S.append(t1)



#initialize vans here
numberOfVans = 1 # set to 30 later
van = []
for i in range(numberOfVans):
    van.append(Van(random.randint(0, 9)))

#initialize list of people
people = []

runTime = 20 #number of ticks, increase later
#ticks script here - ticks * 4 to accomodate for animation
for tick in range(runTime * 4):
    if (tick % 16 == 0):
        # insert scripts for every 4 clock ticks here
        print("4thTick: " + str(tick / 16))
        assignVan(people, van)
        for x in range(numberOfVans):
            if len(van[x].R) > 0:  # check contents of R, if not empty, schedule
                pickupSchedule(van[x])
    if(tick % 4 == 0): # every 4 "ticks" = 1 clock tick
        #insert scripts for every tick here
        generatePeople(people)
        for x in range(numberOfVans):
            print("Van Location: ", van[x].currentNode)

            if len(van[x].S) > 0: # if location scheduled in S, set nextNode to next in path
                if (van[x].S[0].dLocation == van[x].currentNode and van[x].S[0].inVan == True and van[x].mid == False):
                    print("Person at ", van[x].currentNode, " dropped off.")
                    van[x].S.pop(0)
                if (van[x].S[0].pLocation == van[x].currentNode and van[x].S[0].inVan == False and van[x].mid == False):
                    van[x].S[0].inVan = True
                    print("Person at ", van[x].currentNode, " picked up.")
                    dropScheduler(van[x]) # schedule dropoff
                # set nextnode as first element in path from current node to first S location
                print("Person Location: ", van[x].S[0].pLocation, " , ", van[x].S[0].dLocation)
                if (van[x].S[0].inVan == False and van[x].mid == False and van[x].currentNode != van[x].S[0].pLocation):
                    van[x].nextNode = nx.astar_path(G, van[x].currentNode, van[x].S[0].pLocation)[1]
                elif (van[x].S[0].inVan == True and van[x].mid == False and van[x].currentNode != van[x].S[0].dLocation):
                    van[x].nextNode = nx.astar_path(G, van[x].currentNode, van[x].S[0].dLocation)[1]
                if van[x].nextNode != van[x].currentNode and van[x].mid == False: # when van is 1 mile away, go halfway
                    van[x].mid = True
                    van[x].currentNode = van[x].nextNode
                elif van[x].nextNode == van[x].currentNode and van[x].mid == True:
                    van[x].mid = False
                if (van[x].S[0].dLocation == van[x].currentNode and van[x].S[0].inVan == True and van[x].mid == False):
                    print("Person at ", van[x].currentNode, " dropped off.")
                    van[x].S.pop(0)
                if (van[x].S[0].pLocation == van[x].currentNode and van[x].S[0].inVan == False and van[x].mid == False):
                    van[x].S[0].inVan = True
                    print("Person at ", van[x].currentNode, " picked up.")
                    dropScheduler(van[x]) # schedule dropoff
        print("Tick: " + str(tick / 4))

    #animation scripts here
    # records each location of van for each tick
    for i in range(numberOfVans):
        van[i].xyData.append(anim_lerp(van[i].currentNode, van[i].nextNode, ((tick % 8)/4), pos))

### ANIMATION
fig = plt.gcf()
ax = fig.gca()

graphVans = {}
for n in range(numberOfVans):
    graphVans[n] = plt.Circle((van[n].xyData[0]), .05, zorder=10)

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

#anim = FuncAnimation(fig, func=anim_run, init_func=anim_init, frames=runTime * 4, interval=150)

plt.show()

