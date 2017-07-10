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

class CoFIM():
    def __init__(self, graph_path, comm_path):
        self.graph, self.num_node = self.load_graph(graph_path)
        self.comm = self.load_comm(comm_path)

    def load_graph(self, graph_path):
        G = nx.Graph()
        with open(graph_path,'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    num_node, num_edge = line.strip().split('\t')
                    continue
                node1,node2=line.strip().split('\t')
                G.add_edge(int(node1),int(node2))
        return G, G.number_of_nodes()

    def load_comm(self, comm_path):
        node2comm = dict()
        with open(comm_path,'r') as f:
            for i, line in enumerate(f):
                # node,commNo= line.strip().split('\t')
                # node2comm[int(node)-1]=int(commNo)-1
                nodes_in_comm = line.strip().split('\t')
                for node in nodes_in_comm:
                    node2comm[int(node)] = i
        return node2comm

    def get_score(self,seed,gamma):
        neigh_node=self.graph.neighbors(seed)
        neigh_comm=set()
        for node in neigh_node:
            neigh_comm.add(self.comm[node])
        return len(neigh_node)+len(neigh_comm)*gamma

    def marginal_gain(self,neigh_node,neigh_comm,node,gamma):
        tmp_node=set()
        tmp_comm=set()
        for item in self.graph.neighbors(node):
            comm=self.comm[item]
            if item not in neigh_node:
                tmp_node.add(item)
            if comm not in neigh_comm:
                tmp_comm.add(comm)
        return len(tmp_node)+len(tmp_comm)*gamma

    def add_seed(self,seed_set,neigh_node,neigh_comm,node):
        for item in self.graph.neighbors(node):
            comm=self.comm[item]
            neigh_node.add(item)
            neigh_comm.add(comm)
        seed_set.append(node)
        return seed_set,neigh_node,neigh_comm

    def seed_selection(self,k,gamma):
        print "Finding top",k,"nodes"
        print "No.\tNode_id"
        avg_degree = 2*self.graph.number_of_edges()/self.graph.number_of_nodes()
        pairs=dict()
        for node in self.graph.nodes():
            if self.graph.degree(node)<avg_degree:
                continue
            score = self.get_score(node,gamma)
            tmp=sorted(pairs.items(),key=lambda item:item[1],reverse=False)
            if len(pairs)>=10*k and score<=tmp[0][1]:
                continue
            pairs[node]=score
            if len(pairs)>10*k:
                pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=False)[1:])
        pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True))
        updated=[True]*self.num_node
        seed_set=[]
        neigh_node=set()
        neigh_comm=set()
        #total_score=0.0
        for i in xrange(0,k):
            best_pair = sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
            pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            while(updated[int(best_pair[0])] is not True):
                m_gain=self.marginal_gain(neigh_node,neigh_comm,best_pair[0],gamma)
                updated[best_pair[0]]=True
                pairs[best_pair[0]]=m_gain
                best_pair=sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
                pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            seed_set,neigh_node,neigh_comm=self.add_seed(seed_set,neigh_node,neigh_comm,best_pair[0])
            # total_score+=best_pair[1]
            print i+1,"\t",best_pair[0]
            updated=[False]*self.num_node
        return seed_set



    def draw_graph(self):
        nx.draw(self.graph)
        plt.show()

if __name__ == '__main__':
    # network='../weighted_directed_nets/network.dat'
    # community='../weighted_directed_nets/community.dat'
    network='NetHEHT.txt'
    community='NetHEHT_com.txt'
    cofim = CoFIM(network,community)
    seeds=cofim.seed_selection(50,3)
    a=[50]
    for num in a:
        inf=Evaluation.mc_method1(cofim,seeds,num,1000)
        print "seed number:",num,"Total influence:",inf
