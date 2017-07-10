#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import matplotlib.pyplot as plt
#import graphviz
from Queue import PriorityQueue
import random
import Evaluation
from priorityQueue import PriorityQueue as PQ # priority queue


class Degree_Discount():
    def __init__(self, graph_path):
        self.graph, self.num_node = self.load_graph(graph_path)

    def load_graph(self, graph_path):
        G = nx.Graph()
        with open(graph_path,'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    num_node, num_edge = line.strip().split('\t')
                    continue
                node1,node2=line.strip().split('\t')
                G.add_edge(int(node1),int(node2))
        num_node=G.number_of_nodes()
        return G,num_node# int(num_node), int(num_edge)


    def degreeDiscountIC(self, k, p=.01):
        ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
        Input: G -- networkx graph object
        k -- number of nodes needed
        p -- propagation probability
        Output:
        S -- chosen k nodes
        '''
        S = []
        # dd = PQ() # degree discount
        t = dict() # number of adjacent vertices that are in S
        dd = dict() # degree of each vertex

    # initialize degree discount
        for u in self.graph.nodes():
            dd[u] = self.graph.degree(u) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        #     dd.add_task(u, -d[u]) # add degree of each node
            t[u] = 0

    # add vertices to S greedily
        print "Finding top",k,"nodes"
        print "No.\tNode_id"
        for i in range(k):
            best_pair=sorted(dd.items(),key=lambda item:item[1],reverse=True)[0]
            # u, priority = dd.pop_item() # extract node with maximal degree discount
            S.append(best_pair[0])
            print i+1,"\t",best_pair[0]
            for v in self.graph.neighbors(best_pair[0]):
                if v not in S:
                    t[v] += 1 # increase number of selected neighbors
                    # priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                    # dd.add_task(v, -priority)
                    dv=self.graph.degree(v)
                    dd[v]=dv-2*t[v]-(dv-t[v])*t[v]*p
            del dd[best_pair[0]]
        return S



    def draw_graph(self):
        nx.draw(self.graph)
        plt.show()

if __name__ == '__main__':
    DD = Degree_Discount('../weighted_directed_nets/network.dat')
    seeds=DD.degreeDiscountIC(50)
    # SD2 = Single_Discount('NetHEHT.txt')
    # inf=Evaluation.monte_carlo_extend(DD,list(seeds),50,100)
    # print "Total influence:",inf
    a=[1,5,10,15,20,25,30,35,40,45,50]
    for num in a:
        inf=Evaluation.monte_carlo(DD,seeds,num,10000)
        print "seed number:",num,"Total influence:",inf