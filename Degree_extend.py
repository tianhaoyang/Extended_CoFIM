#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import matplotlib.pyplot as plt
#import graphviz
from Queue import PriorityQueue
import random
import Evaluation


class Degree_Heuristic():
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
        return G,num_node#, int(num_node), int(num_edge)


    def seed_selection(self,k):
        print "Finding top",k,"nodes"
        print "No.\tNode_id\tDegree\tTimes(s)"
        pairs=dict()
        seed_set=[]
        for node in self.graph.nodes():
            degree = self.graph.out_degree(node)
            pairs[node]=degree
            # if len(pairs)>k:
            #     pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=False)[1:])
        # s=sorted(pairs.items(),key=lambda item:item[1],reverse=True)
        for i in xrange(0,k):

            best_pair = sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
            pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            seed_set.append(best_pair[0])
            # print i+1,"\t",s[i][0],"\t",s[i][1],"\t",0
            print i+1,"\t",best_pair[0],"\t",best_pair[1],"\t",0
        return seed_set


    def draw_graph(self):
        nx.draw(self.graph)
        plt.show()

if __name__ == '__main__':
    DH = Degree_Heuristic('../weighted_directed_nets/network.dat')
    # deg=0.0
    # for node in DH.graph.nodes():
    #     if deg<DH.graph.in_degree(node):
    #         deg=DH.graph.in_degree(node)
    # print DH.graph.number_of_nodes()
    seeds=DH.seed_selection(50)
    a=[1,5,10,15,20,25,30,35,40,45,50]
    for num in a:
        inf=Evaluation.mc_method(DH,seeds,num,100)
        print "seed number:",num,"Total influence:",inf
