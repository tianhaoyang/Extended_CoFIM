#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import matplotlib.pyplot as plt
#import graphviz
from Queue import PriorityQueue
import random
import Evaluation


class Single_Discount():
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
        return G, num_node#, int(num_edge)


    def seed_selection(self,k):
        print "Finding top",k,"nodes"
        print "No.\tNode_id\tDegree"
        seed_set=[]

        for i in xrange(0,k):
            degree_dict=self.graph.out_degree(self.graph.nodes())
            max_degree = sorted(degree_dict.items(),key=lambda item:item[1],reverse=True)[0]
            seed_set.append(max_degree[0])
            print i+1,"\t",max_degree[0],"\t",max_degree[1]
            self.graph.remove_node(max_degree[0])
        return seed_set


    def draw_graph(self):
        nx.draw(self.graph)
        plt.show()

if __name__ == '__main__':
    network='../weighted_directed_nets/network.dat'
    SD = Single_Discount(network)
    seeds=SD.seed_selection(50)
    SD2 = Single_Discount(network)
    a=[1,5,10,15,20,25,30,35,40,45,50]
    for num in a:
        inf=Evaluation.mc_method(SD2,seeds,num,100)
        print "seed number:",num,"Total influence:",inf
