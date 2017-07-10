#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import matplotlib.pyplot as plt
#import graphviz
from Queue import PriorityQueue
import random
import Evaluation

# S_STATE=0
# I_STATE=1
# SI_STATE=2
# R_STATE=3

class CELF():
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

    def seed_selection(self,k,gamma):
        print "Finding top",k,"nodes"
        print "No.\tNode_id\tTimes(s)"
        pairs=dict()
        seed_set=[]
        for node in self.graph.nodes():
            seed_set.append(node)
            inf = Evaluation.monte_carlo(self,seed_set,1,10)
            pairs[node]=inf
            seed_set.remove(node)
        updated=[True]*self.num_node
        #total_score=0.0
        for i in xrange(0,k):
            best_pair = sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
            pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            while(updated[int(best_pair[0])] is not True):
                seed_set.append(best_pair[0])
                inf = Evaluation.monte_carlo(self,list(seed_set),i+1,10)
                seed_set.remove(best_pair[0])
                pairs[best_pair[0]]=inf
                updated[best_pair[0]]=True
                best_pair=sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
                pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            seed_set.append(best_pair[0])
            # total_score+=best_pair[1]
            print i+1,"\t",best_pair[0],"\t",0
            updated=[False]*self.num_node
        return seed_set



    def draw_graph(self):
        nx.draw(self.graph)
        plt.show()

if __name__ == '__main__':
    celf = CELF('NetHEHT.txt')
    seeds=celf.seed_selection(50,3)
    # inf=Evaluation.monte_carlo(celf,list(seeds),50,10)
    # print "Total influence:",inf
    a=[1,5,10,15,20,25,30,35,40,45,50]
    for num in a:
        inf=Evaluation.monte_carlo(celf,seeds,num,10000)
        print "seed number:",num,"Total influence:",inf