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
        self.graph, self.num_node, self.num_edge = self.load_graph(graph_path)

    def load_graph(self, graph_path):
        G = nx.Graph()
        with open(graph_path,'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    num_node, num_edge = line.strip().split('\t')
                    continue
                node1,node2=line.strip().split('\t')
                G.add_edge(int(node1),int(node2))
        return G, int(num_node), int(num_edge)


    def seed_selection(self,k):
        print "Finding top",k,"nodes"
        print "No.\tNode_id\tDegree\tTimes(s)"
        seed_set=[]

        for i in xrange(0,k):
            degree_dict=self.graph.degree(self.graph.nodes())
            max_degree = sorted(degree_dict.items(),key=lambda item:item[1],reverse=True)[0]
            seed_set.append(max_degree[0])
            print i+1,"\t",max_degree[0],"\t",max_degree[1],"\t",0
            self.graph.remove_node(max_degree[0])
        return seed_set


    def draw_graph(self):
        nx.draw(self.graph)
        plt.show()

if __name__ == '__main__':
    SD = Single_Discount('NetHEHT.txt')
    seeds=SD.seed_selection(50)
    SD2 = Single_Discount('NetHEHT.txt')
    a=[1,5,10,15,20,25,30,35,40,45,50]
    for num in a:
        inf=Evaluation.monte_carlo(SD2,seeds,num,10000)
        print "seed number:",num,"Total influence:",inf
