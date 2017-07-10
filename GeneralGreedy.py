#coding:utf8
''' Implements greedy heuristic for IC model [1]
[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 1)
'''
__author__ = 'ivanovsergey'

from priorityQueue import PriorityQueue as PQ
from IC import runIC
import networkx as nx
import Evaluation

def load_graph(graph_path,directed,weighted):
    G = nx.DiGraph()
    with open(graph_path,'r') as f:
        if weighted is True:
            for i,line in enumerate(f):
                node1,node2,weight=line.strip().split('\t')
                G.add_edge(int(node1)-1,int(node2)-1,weight=float(weight))
                if directed is False:
                    G.add_edge(int(node2)-1,int(node1)-1,weight=float(weight))
        else:
            for i,line in enumerate(f):
                if i==0:
                    continue
                node1,node2=line.strip().split('\t')
                G.add_edge(int(node1),int(node2),weight=1.0)
                if directed is False:
                    G.add_edge(int(node2),int(node1),weight=1.0)
                # nx.set_node_attributes(G,'Second_round',1)
    num_node=G.number_of_nodes()
    return G, num_node#, int(num_edge)

def generalGreedy(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    import time
    start = time.time()
    R = 20 # number of times to run Random Cascade
    S = [] # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):
        s = PQ() # priority queue
        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0) # initialize spread value
                for j in range(R): # run R times Random Cascade
                    [priority, count, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(runIC(G, S + [v], p)))/R) # add normalized spread value
        task, priority = s.pop_item()
        S.append(task)
        print i+1, task, time.time() - start
    return S

if __name__=='__main__':
    G, nodes =load_graph('NetHEHT.txt',False,False)
    S=generalGreedy(G,k=10)
    inf=Evaluation.monte_carlo_extend3(G,S,k=10,num_simu=100)
    print "Seeds number:",50,"Total influence:",inf