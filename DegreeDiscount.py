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
        G = nx.DiGraph()
        with open(graph_path,'r') as f:
            for i, line in enumerate(f):
                # if i == 0:
                #     num_node, num_edge = line.strip().split('\t')
                #     continue
                node1,node2,weight=line.strip().split('\t')
                G.add_edge(int(node1)-1,int(node2)-1,weight=float(weight))
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
        dd = PQ() # degree discount
        t = dict() # number of adjacent vertices that are in S
        d = dict() # degree of each vertex

    # initialize degree discount
        for u in self.graph.nodes():
            d[u] = sum([self.graph[u][v]['weight'] for v in self.graph.successors(u)]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
            dd.add_task(u, -d[u]) # add degree of each node
            t[u] = 0

    # add vertices to S greedily
        for i in range(k):
            u, priority = dd.pop_item() # extract node with maximal degree discount
            S.append(u)
            for v in self.graph[u]:
                if v not in S:
                    t[v] += self.graph[u][v]['weight'] # increase number of selected neighbors
                    priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                    dd.add_task(v, -priority)
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
        inf=Evaluation.mc_method(DD,seeds,num,100)
        print "seed number:",num,"Total influence:",inf