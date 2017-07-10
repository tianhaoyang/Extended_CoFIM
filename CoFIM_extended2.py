#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#import graphviz
from Queue import PriorityQueue
import random
import Evaluation

# S_STATE=0
# I_STATE=1
# SI_STATE=2
# R_STATE=3

class extended_CoFIM():
    def __init__(self, graph_path, comm_path,alpha,budget):
        self.graph,self.num_node = self.load_graph(graph_path)
        self.comm ,self.comm2node= self.load_comm(comm_path)
        # self.graph=self.weight_adjust(self.graph,alpha)
        self.pagerank=nx.pagerank(self.graph)
        self.alpha=alpha
        self.budget=budget

    def load_graph(self, graph_path):
        G = nx.DiGraph()
        with open(graph_path,'r') as f:
            for line in f:
                # if i == 0:
                #     num_node, num_edge = line.strip().split('\t')
                #     continue
                node1,node2,weight=line.strip().split('\t')
                G.add_edge(int(node1)-1,int(node2)-1,weight=float(weight))
                # nx.set_node_attributes(G,'Second_round',1)
        num_node=G.number_of_nodes()
        return G, num_node#, int(num_edge)

    def load_comm(self, comm_path):
        node2comm = dict()
        comm2node=dict()
        with open(comm_path,'r') as f:
            for i, line in enumerate(f):
                node,commNo= line.strip().split('\t')
                node=int(node)-1
                commNo=int(commNo)-1
                node2comm[node]=commNo
                if commNo not in comm2node.keys():
                    comm2node[commNo]=[node]
                else:
                    comm2node[commNo].append(node)
                # for node in nodes_in_comm:
                #     node2comm[int(node)] = i
        return node2comm,comm2node

    def weight_adjust(self,graph,alpha):
        for node in graph.nodes():
            total=0
            count=0
            for pre in graph.predecessors(node):
                if self.comm[pre] is not self.comm[node]:
                    total+=graph[pre][node]['weight']
                else:
                    count+=1
            for pre in graph.predecessors(node):
                if self.comm[pre] is not self.comm[node]:
                    graph[pre][node]['weight']*=(1-alpha)
                else:
                    graph[pre][node]['weight']+=alpha*total/count
        return graph

    def struc_cost(self,node,lamda,delta):
        suc=self.graph.successors(node)
        list(suc).append(node)
        max_deg=max(self.graph.out_degree(i) for i in suc)
        # return (lamda*(self.pagerank[node]+delta)*self.graph.out_degree(node) )/max_deg
        return self.graph.out_degree(node)

    def get_score(self,seed,gamma):
        successors=self.graph.successors(seed)
        # second_order=0
        # prob=[]
        # sum_weight=sum(self.graph[seed][suc]['weight'] for suc in self.graph.successors(seed))
        # for suc in self.graph.successors(seed):
        #
        #
        #     prob.append(1.0-self.graph[seed][suc]['weight']/sum_weight)
        #     prob=1.0-1.0/self.graph.in_degree(suc)
        #     second_order+=(1.0-prob)
        # list(successors).append(seed)
        # inf = Evaluation.monte_carlo_extend(self,successors)
        tmp_comm=set()
        for node in successors:
            tmp_comm.add(self.comm[node])
        intra_comm=0
        for comm in tmp_comm:
            intra_comm+=0.01*len(self.comm2node[comm])
        self.graph.node[seed]['hard_eff']=0
        self.graph.node[seed]['easy_eff']=0
        # self.graph.node[seed]['else']=0
        # return (second_order+len(neigh_comm)*10)#/self.struc_cost(seed,100,0.5)
        # return second_order+intra_comm
        # return second_order+inf
        # return len(successors)+len(tmp_comm)*10
        return len(successors)+intra_comm

    def marginal_gain(self,seed_set,neigh_node,neigh_comm,node,gamma):
        inf1=0
        inf2=0
        # total=0

        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        count6=0
        count7=0
        count8=0
        for suc in list(neigh_node):

            prod=1
            for pre in self.graph.predecessors(suc):
                if pre in seed_set:
                    sum_weight=sum(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    prod*=(1.0-self.graph[pre][suc]['weight']/sum_weight)
                    # total=sum(self.graph.out_degree(suc) for suc in )
                    # if self.comm[suc]==self.comm[pre]:
                    #     p=self.graph[pre][suc]['weight']/sum_weight*(1.0+self.alpha)
                    # else:
                    #     p=self.graph[pre][suc]['weight']/sum_weight*(1.0-self.alpha)
                    # prod*=(1.0-p)
                    # prod*=(1.0-1.0/self.graph.in_degree(suc))
                    # count1+=1
            # inf1+=(1.0-prod)
            # print inf1
            prod2=1
            for pre in self.graph.predecessors(suc):
                if pre in seed_set:
                    sum_weight=sum(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    max_weight=max(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    max_degree=max(self.graph.out_degree(suc) for suc in self.graph.successors(pre))

                    if self.graph[pre][suc]['weight']<=0.5*max_weight and self.graph.out_degree(suc)>0.9*max_degree:
                        prod2*=(1.0-self.graph[pre][suc]['weight']/sum_weight)
                        count1+=1
                    elif self.graph[pre][suc]['weight']>0.5*max_weight and self.graph.out_degree(suc)>0.9*max_degree:
                        prod2*=(1.0-self.graph[pre][suc]['weight']/sum_weight)
                        count3+=1
                    # else:
                    #     count5+=1
                        # if self.comm[suc]==self.comm[pre]:
                        #     p=self.graph[pre][suc]['weight']/sum_weight*(1.0+self.alpha)
                        # else:
                        #     p=self.graph[pre][suc]['weight']/sum_weight
                        # prod2*=(1.0-p)
            inf1+=(1.0-prod)+prod*(1.0-prod2)
            # print inf1
        # print inf1


        # print len(neigh_node)
        # l=len(neigh_comm)
        # count2=0
        tmp_node=list(neigh_node)
        tmp_seed=list(seed_set)
        tmp_comm=set()
        for item in self.graph.successors(node):
            # neigh_comm.add(self.comm[item])
            if item not in tmp_node:
                tmp_node.append(item)
            comm=self.comm[item]
            if comm not in neigh_comm:
                tmp_comm.add(comm)
        tmp_seed.append(node)
        # print len(neigh_node)
        for suc in tmp_node:

            prod2=1
            for pre in self.graph.predecessors(suc):
                if pre in tmp_seed:
                    sum_weight=sum(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    prod2*=(1.0-self.graph[pre][suc]['weight']/sum_weight)
                    # if self.comm[suc]==self.comm[pre]:
                    #     p=self.graph[pre][suc]['weight']/sum_weight*(1.0+self.alpha)
                    # else:
                    #     p=self.graph[pre][suc]['weight']/sum_weight
                    # prod2*=(1.0-p)
                    # prod2*=(1.0-1.0/self.graph.in_degree(suc))
                    # count2+=1
            # inf2+=(1.0-prod2)
            prod3=1
            flag=0
            for pre in self.graph.predecessors(suc):
                if pre in tmp_seed:
                    sum_weight=sum(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    max_weight=max(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    max_degree=max(self.graph.out_degree(suc) for suc in self.graph.successors(pre))

                    if self.graph[pre][suc]['weight']<=0.5*max_weight and self.graph.out_degree(suc)>0.8*max_degree:
                        prod3*=(1.0-self.graph[pre][suc]['weight']/sum_weight)
                        count2+=1
                    elif self.graph[pre][suc]['weight']>0.5*max_weight and self.graph.out_degree(suc)>0.8*max_degree:
                        prod3*=(1.0-self.graph[pre][suc]['weight']/sum_weight)
                        count4+=1
                    # else:
                    #     count6+=1
                        # if self.comm[suc]==self.comm[pre]:
                        #     p=self.graph[pre][suc]['weight']/sum_weight*(1.0+self.alpha)
                        # else:
                        #     p=self.graph[pre][suc]['weight']/sum_weight
                        # prod3*=(1.0-p)


            inf2+=(1.0-prod2)+prod2*(1.0-prod3)


        self.graph.node[node]['hard_eff']=count2-count1
        self.graph.node[node]['easy_eff']=count4-count3
        # self.graph.node[node]['else']=count6-count5
        # print inf2
        #
        # tmp_node=set()
        # tmp_comm=set()
        # for item in self.graph.successors(node):
        #     comm=self.comm[item]
        #     # if item not in neigh_node:
        #     #     tmp_node.add(item)
        #     if comm not in neigh_comm:
        #         tmp_comm.add(comm)
        # seed_set.add(node)
        # inf=0
        # for suc in list(tmp_node):
        #     prod=[]
        #     sum_weight=sum(self.graph[pre][suc]['weight'] for pre in self.graph.predecessors(suc))
        #     for pre in self.graph.predecessors(suc):
        #         if pre in seed_set:
        #             prod.append(1.0-self.graph[pre][suc]['weight']/sum_weight)
        #     inf+=(1-np.prod(prod))
        # inf1=Evaluation.monte_carlo_extend(self,list(seed_set))
        # seed_set.add(node)
        # inf2=Evaluation.monte_carlo_extend(self,list(seed_set))
        # print len(neigh_comm2)-len(neigh_comm)
        # print inf2-inf1
        # print len(tmp_comm)
        intra_comm=0
        for comm in tmp_comm:
            intra_comm+=0.01*len(self.comm2node[comm])
        # return ((inf2-inf1)+(len(tmp_comm))*10)#/self.struc_cost(node,100,0.5)
        # return len(tmp_node)+len(tmp_comm)*10
        return (inf2-inf1)+intra_comm

    def add_seed(self,seed_set,neigh_node,neigh_comm,node,total_cost):
        for item in self.graph.successors(node):
            comm=self.comm[item]
            neigh_node.add(item)
            neigh_comm.add(comm)
        seed_set.add(node)
        total_cost+=self.struc_cost(node,50,0.5)
        # print self.struc_cost(node,10,0.5)
        total_cost+=4*self.graph.node[node]['hard_eff']
        total_cost+=2*self.graph.node[node]['easy_eff']
        # total_cost+=1*self.graph.node[node]['else']
        return seed_set,neigh_node,neigh_comm,total_cost

    def node_expansion(self,k,gamma):
        print "Finding top",k,"nodes"
        print "No.\tNode_id\tTimes(s)"
        avg_degree=0
        for node in self.graph.nodes():
            avg_degree+=self.graph.out_degree(node)
        avg_degree = avg_degree/self.graph.number_of_nodes()
        pairs=dict()
        for node in self.graph.nodes():
            if self.graph.out_degree(node)<avg_degree:
                continue
            score = self.get_score(node,gamma)
            tmp=sorted(pairs.items(),key=lambda item:item[1],reverse=False)
            if len(pairs)>=10*k and score<=tmp[0][1]:
                continue
            pairs[node]=score
            if len(pairs)>10*k:
                pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=False)[1:])
        pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True))
        updated=list()
        for i in xrange(0,self.graph.number_of_nodes()):
            updated.append(True)
        seed_set=set()
        neigh_node=set()
        neigh_comm=set()
        total_score=0.0
        total_cost=0.0
        for i in xrange(0,k):
            best_pair = sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
            pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            while(updated[int(best_pair[0])] is not True):
                m_gain=self.marginal_gain(seed_set,neigh_node,neigh_comm,best_pair[0],gamma)
                updated[best_pair[0]]=True
                pairs[best_pair[0]]=m_gain
                best_pair=sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
                pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            seed_set,neigh_node,neigh_comm,total_cost=self.add_seed(seed_set,neigh_node,neigh_comm,best_pair[0],total_cost)
            if total_cost>=self.budget:
                break
            total_score+=best_pair[1]
            print i+1,"\t",best_pair[0],"\t",0,self.budget-total_cost
            for i in xrange(0,self.graph.number_of_nodes()):
                updated[i]=False
            # print(len(seed_set))
        return seed_set

    # def monte_carlo(self,seed,k,num_simu=10000):
    #     random.seed()
    #     inf=0.0
    #     i_size=0
    #     r_size=0
    #     si_size=0
    #     i_arr=[0]*self.num_node
    #     r_arr=[0]*self.num_node
    #     si_arr=[0]*self.num_node
    #     state=[S_STATE]*self.num_node
    #     for r in xrange(0,num_simu):
    #         active_size=0
    #         for i in xrange(0,r_size):
    #             state[r_arr[i]]=S_STATE
    #         r_size=0
    #
    #         for i in xrange(0,k):
    #             i_arr[i_size]=seed[i]
    #             state[i_arr[i]]=I_STATE
    #             i_size+=1
    #
    #         while(i_size>0):
    #             active_size+=i_size
    #             si_size=0
    #             for i in xrange(0,i_size):
    #                 for node in self.graph.neighbors(i_arr[i]):
    #                     if state[node]==S_STATE:
    #                     # if state[node]==S_STATE and self.comm[node]==self.comm[i_arr[i]]:
    #                         pp=1.0/self.graph.degree(node)
    #                         rand=random.random()
    #                         if rand<pp:
    #                             state[node]=SI_STATE
    #                             si_arr[si_size]=node
    #                             si_size+=1
    #             for i in xrange(0,i_size):
    #                 state[i_arr[i]]=R_STATE
    #                 r_arr[r_size]=i_arr[i]
    #                 r_size+=1
    #             i_size=0
    #             for i in xrange(0,si_size):
    #                 state[si_arr[i]]=I_STATE
    #                 i_arr[i_size]=si_arr[i]
    #                 i_size+=1
    #         inf+=active_size
    #     print "Total influence:",inf/num_simu
    def write_seeds(self,seeds):
        with open('seeds.txt','w') as f:
            for seed in list(seeds):
                f.write(str(seed))
                f.write('\n')

    def load_seeds(self,seeds_path):
        seeds=[]
        with open(seeds_path,'r') as f:
            for seed in f.read():
                seeds.append(seed)
        return seeds



    def draw_graph(self):
        pos=nx.spring_layout(self.graph)
        edge_labels=dict([((u,v,),d['weight']) for u,v,d in self.graph.edges(data=True)])
        nx.draw_networkx_edge_labels(self.graph,pos,edge_labels=edge_labels)
        nx.draw(self.graph,pos=pos,node_size=100,arrows=True)
        plt.show()

if __name__ == '__main__':
    excofim = extended_CoFIM('../weighted_directed_nets/network.dat','../weighted_directed_nets/community.dat',0,1000)
    seeds=excofim.node_expansion(50,3)
    print seeds
    inf=Evaluation.monte_carlo_extend2(excofim,list(seeds),num_simu=100)
    print "Total influence:",inf
    # excofim.draw_graph()
