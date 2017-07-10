#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
#import graphviz
from Queue import PriorityQueue
import random
import Evaluation
import Intra_comm
import Predictor
import pathos
import pathos.multiprocessing as mp
import multiprocessing
import pp
import sys
import time
import platform
from psutil import virtual_memory

class extended_CoFIM():
    def __init__(self, graph_path, comm_path,alpha,budget,directed,weighted,parallel):
        self.graph,self.num_node = self.load_graph(graph_path,directed,weighted)
        self.comm,self.comm2node= self.load_comm(comm_path)
        self.classifiers,self.comm_deg_centra,self.comm_between_centra,self.comm_load_centra,\
        self.comm_avg_nei_deg,self.comm_harmonic_centra,self.comm_close_centra=self.communities(parallel)
        self.exscore=self.external_score()
        self.alpha=alpha
        self.budget=budget

    def load_graph(self, graph_path,directed,weighted):
        print "Loading graph..."
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
                    # if i==0:
                    #     continue
                    node1,node2=line.strip().split('\t')
                    G.add_edge(int(node1),int(node2),weight=float(1.0))
                    if directed is False:
                        G.add_edge(int(node2),int(node1),weight=float(1.0))
                # nx.set_node_attributes(G,'Second_round',1)
        num_node=G.number_of_nodes()
        return G, num_node#, int(num_edge)

    def load_comm(self, comm_path):
        print "Loading communities..."
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
                # nodes_in_comm = line.strip().split('\t')
                # comm2node[i]=[]
                # for node in nodes_in_comm:
                #     comm2node[i].append(int(node))
                # for node in nodes_in_comm:
                #     node2comm[int(node)] = i

        return node2comm,comm2node

    def communities(self,parallel):
        print "Training classifiers..."
        # multiprocessing.freeze_support()
        # cores=mp.cpu_count()
        # cores = multiprocessing.cpu_count()
        # pool = mp.ProcessingPool(4)
        # subgraphs={}
        classifiers={}
        deg_centra={}
        between_centra={}
        load_centra={}
        avg_nei_deg={}
        harmonic_centra={}
        close_centra={}
        # score={}
        # for i,y in enumerate(pool.imap(self.get_classifiers,self.comm2node.keys())):
        #     print i
            # subgraph=y[0]
            # classifiers[i]=y[1]
            # deg_centra[i],between_centra[i],load_centra[i],avg_nei_deg[i],harmonic_centra[i],close_centra[i]=self.attributes(y[0])
            # score=dict(score,**y[2])
        if parallel==False:
            start_time=time.time()
            for comm in self.comm2node.keys():
                print comm
                # nodes=self.comm2node[comm]
                subgraph = self.graph.subgraph(self.comm2node[comm])
                # subgraphs[comm]=subgraph
                classifiers[comm] = Predictor.training(subgraph)
                deg_centra[comm], between_centra[comm], load_centra[comm], avg_nei_deg[comm], harmonic_centra[comm], close_centra[
                    comm] = self.attributes(subgraph)
            print 'non-parallel:', time.time() - start_time, 's'
        else:
            ppservers=()
            if len(sys.argv) > 1:
                ncpus = int(sys.argv[1])
                # Creates jobserver with ncpus workers
                job_server = pp.Server(ncpus, ppservers=ppservers)
            else:
                # Creates jobserver with automatically detected number of workers
                job_server = pp.Server(ppservers=ppservers)

            print "pp 可以用的工作核心线程数", job_server.get_ncpus(), "workers"
            # comms=list(self.comm2node.keys())
            start_time=time.time()
            jobs = [(comm, job_server.submit(self.get_classifiers,(comm,),(),("Predictor",))) for comm in self.comm2node.keys()]
            # print "yes"
            for comm, job in jobs:
                print comm
                classifiers[comm]=job()[1]
                deg_centra[comm],between_centra[comm],load_centra[comm],\
                avg_nei_deg[comm],harmonic_centra[comm],close_centra[comm]=self.attributes(job()[0])
            print 'parallel:', time.time() - start_time, 's'


        return classifiers,deg_centra,between_centra,load_centra,avg_nei_deg,harmonic_centra,close_centra

    def get_classifiers(self,comm):
        # subgraphs={}
        # classifiers={}
        # import Predictor
        # import pathos
        # print pathos.helpers.mp.current_process()
        # print comm
        subgraph=self.graph.subgraph(self.comm2node[comm])
        classifier=Predictor.training(subgraph)


        return subgraph,classifier

    def external_score(self):
        print "----------------"
        score={}
    #     # cores=mp.cpu_count()
    #     # pool=mp.ProcessingPool(cores)
    #     # for y in pool.imap(self.single_comm_external,self.comm2node.keys()):
    #     #     score=dict(score,**y)
        for comm in self.comm2node.keys():
            new_comm={}
            for node in self.comm2node[comm]:
                for neighbour in self.graph.successors(node):
                    nei_comm=self.comm[neighbour]
                    if nei_comm != self.comm[node]:
                        if nei_comm not in new_comm.keys():
                            new_comm[nei_comm]=[node]
                        else:
                            new_comm[nei_comm].append(node)
            for node in self.comm2node[comm]:
                score[node]=0.0
                for neighbour in self.graph.successors(node):
                    nei_comm=self.comm[neighbour]
                    if nei_comm in new_comm.keys():

                        X=np.array([self.comm_deg_centra[nei_comm][neighbour] ,
                                self.comm_between_centra[nei_comm][neighbour] ,
                                self.comm_load_centra[nei_comm][neighbour] ,
                                self.comm_avg_nei_deg[nei_comm][neighbour],
                                self.comm_harmonic_centra[nei_comm][neighbour],
                                self.comm_close_centra[nei_comm][neighbour]]).reshape(1,-1)
                        pred=self.classifiers[nei_comm].predict(X)[0]
                        # if pred<0:
                        #     pred=0.0
                        score[node]+=1.0/len(new_comm[nei_comm])*self.graph[node][neighbour]['weight']*pred

        return score

    # def single_comm_external(self,comm):
    #     # print "---------------"
    #     print pathos.helpers.mp.current_process()
    #     new_comm = {}
    #     score={}
    #     for node in self.comm2node[comm]:
    #         for neighbour in self.graph.successors(node):
    #             nei_comm = self.comm[neighbour]
    #             if nei_comm != self.comm[node]:
    #                 if nei_comm not in new_comm.keys():
    #                     new_comm[nei_comm] = [node]
    #                 else:
    #                     new_comm[nei_comm].append(node)
    #     for node in self.comm2node[comm]:
    #         score[node] = 0.0
    #         for neighbour in self.graph.successors(node):
    #             nei_comm = self.comm[neighbour]
    #             if nei_comm in new_comm.keys():
    #                 score[node] += 1.0 / len(new_comm[nei_comm]) * self.graph[node][neighbour]['weight'] * len(
    #                     self.comm2node[nei_comm])
    #                 # deg_centra, between_centra, load_centra, avg_nei_deg = self.attributes(self.subgraphs[nei_comm])
    #                 # print deg_centra
    #                 X = np.array([self.comm_deg_centra[nei_comm][neighbour],
    #                               self.comm_between_centra[nei_comm][neighbour],
    #                               self.comm_load_centra[nei_comm][neighbour],
    #                               self.comm_avg_nei_deg[nei_comm][neighbour],
    #                               self.comm_harmonic_centra[nei_comm][neighbour],
    #                               self.comm_close_centra[nei_comm][neighbour]]).reshape(1, -1)
    #                 score[node] += self.classifiers[nei_comm].predict(X)
    #     return score


    def struc_cost(self,node,lamda=None,delta=None):
        # suc=self.graph.successors(node)
        # list(suc).append(node)
        # max_deg=max(self.graph.out_degree(i) for i in suc)
        # return (lamda*(self.pagerank[node]+delta)*self.graph.out_degree(node) )/max_deg
        return self.graph.out_degree(node)#+10*self.exscore[node]

    def attributes(self,graph):
        if len(graph)==1:
            # print("yes")
            deg_centra={graph.nodes()[0]:1}
        else:
            deg_centra=nx.degree_centrality(graph)
        between_centra=nx.betweenness_centrality(graph)
        load_centra=nx.load_centrality(graph)
        # eigen_centra=nx.eigenvector_centrality(self.graph)
        avg_neigh_deg=nx.average_neighbor_degree(graph)
        harmonic_centra=nx.harmonic_centrality(graph)
        close_centra=nx.closeness_centrality(graph)
        return deg_centra,between_centra,load_centra,avg_neigh_deg,harmonic_centra,close_centra


    def get_score(self,seed,gamma):
        #total_deg=sum(self.graph.out_degree(suc) for suc in self.graph.successors(seed))/self.graph.out_degree(seed)
        # avg_nei=nx.average_neighbor_degree(G=self.graph,nodes=[seed])
        # successors=self.graph.successors(seed)
        inf1=0.0
        for suc in self.graph.successors(seed):

            # prod=1.0
            sum_weight=sum(self.graph[p][suc]['weight'] for p in self.graph.predecessors(suc))
            prod=(1.0-self.graph[seed][suc]['weight']/sum_weight)

            # inf1+=(1.0-prod)
            prod2=1.0
            # sum_weight=sum(self.graph[p][suc]['weight'] for p in self.graph.predecessors(suc))
            max_weight=max(self.graph[seed][suc]['weight'] for suc in self.graph.successors(seed))
            max_degree=max(self.graph.out_degree(suc) for suc in self.graph.successors(seed))
            if self.graph[seed][suc]['weight']>0.7*max_weight or self.graph.out_degree(suc)>0.7*max_degree:
                prod2*=(1.0-self.graph[seed][suc]['weight']/sum_weight)

            inf1+=(1.0-prod)+prod*(1.0-prod2)


        # tmp_comm=set()
        # nodes_in_new_comm={}
        # for node in self.graph.successors(seed):
        #     comm=self.comm[node]
        #     if comm is not self.comm[seed]:
        #         tmp_comm.add(comm)
                # if comm not in nodes_in_new_comm.keys():
                #     nodes_in_new_comm[comm]=[node]
                # else:
                #     nodes_in_new_comm[comm].append(node)
        intra_comm=0.0
        # for comm in nodes_in_new_comm.keys():
            # intra_comm+=0.01*len(self.comm2node[comm])


        # intra_comm=0.0
        for suc in self.graph.successors(seed):
            # if self.comm[suc] !=self.comm[seed]:
            X = np.array([self.comm_deg_centra[self.comm[suc]][suc],
                          self.comm_between_centra[self.comm[suc]][suc],
                          self.comm_load_centra[self.comm[suc]][suc],
                          self.comm_avg_nei_deg[self.comm[suc]][suc],
                          self.comm_harmonic_centra[self.comm[suc]][suc],
                          self.comm_close_centra[self.comm[suc]][suc]]).reshape(1, -1)
            intra_comm+=float(self.classifiers[self.comm[suc]].predict(X)[0])#+self.exscore[suc]

        # X = np.array([self.comm_deg_centra[self.comm[seed]][seed],
        #               self.comm_between_centra[self.comm[seed]][seed],
        #               self.comm_load_centra[self.comm[seed]][seed],
        #               self.comm_avg_nei_deg[self.comm[seed]][seed],
        #               self.comm_harmonic_centra[self.comm[seed]][seed],
        #               self.comm_close_centra[self.comm[seed]][seed]]).reshape(1, -1)
        # intra_comm = float(self.classifiers[self.comm[seed]].predict(X)[0]) #+self.exscore[seed]

        # return self.graph.out_degree(seed)+avg_nei[seed]+self.exscore[seed]
        # print inf1,intra_comm
        return inf1+intra_comm*0.01#+self.exscore[seed]*0.01

    def marginal_gain(self,seed_set,neigh_node,neigh_comm,node,gamma,avg_ext):
        inf1=0
        inf2=0

        for suc in list(neigh_node):
            # if self.exscore[suc]<avg_ext:
            #     continue

            prod=1.0
            for pre in self.graph.predecessors(suc):
                if pre in seed_set:
                    sum_weight=sum(self.graph[p][suc]['weight'] for p in self.graph.predecessors(suc))
                    prod*=(1.0-self.graph[pre][suc]['weight']/sum_weight)

            # inf1+=(1.0-prod)
            prod2=1.0
            for pre in self.graph.predecessors(suc):
                if pre in seed_set:
                    sum_weight=sum(self.graph[p][suc]['weight'] for p in self.graph.predecessors(suc))
                    max_weight=max(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    max_degree=max(self.graph.out_degree(suc) for suc in self.graph.successors(pre))
                    if self.graph[pre][suc]['weight']>0.7*max_weight or self.graph.out_degree(suc)>0.7*max_degree:
                        prod2*=(1.0-self.graph[pre][suc]['weight']/sum_weight)

            inf1+=(1.0-prod)+prod*(1.0-prod2)

        tmp_node=list(neigh_node)
        tmp_seed=list(seed_set)
        # tmp_comm=set()
        # nodes_in_new_comm={}
        more_nodes=[]
        for item in self.graph.successors(node):
            if item not in tmp_node:
                tmp_node.append(item)
                more_nodes.append(item)
            # comm=self.comm[item]
            # if comm not in neigh_comm:
            #     tmp_comm.add(comm)
            #     if comm not in nodes_in_new_comm.keys():
            #         nodes_in_new_comm[comm]=[item]
            #     else:
            #         nodes_in_new_comm[comm].append(item)
        tmp_seed.append(node)

        for suc in tmp_node:
            # if self.exscore[suc]<avg_ext:
            #     continue
            prod2=1.0
            for pre in self.graph.predecessors(suc):
                if pre in tmp_seed:
                    sum_weight=sum(self.graph[p][suc]['weight'] for p in self.graph.predecessors(suc))
                    prod2*=(1.0-self.graph[pre][suc]['weight']/sum_weight)

            # inf2+=(1.0-prod2)
            prod3=1.0
            for pre in self.graph.predecessors(suc):
                if pre in tmp_seed:
                    sum_weight=sum(self.graph[p][suc]['weight'] for p in self.graph.predecessors(suc))
                    max_weight=max(self.graph[pre][suc]['weight'] for suc in self.graph.successors(pre))
                    max_degree=max(self.graph.out_degree(suc) for suc in self.graph.successors(pre))
                    if self.graph[pre][suc]['weight']>0.7*max_weight or self.graph.out_degree(suc)>0.7*max_degree:
                        prod3*=(1.0-self.graph[pre][suc]['weight']/sum_weight)


            inf2+=(1.0-prod2)+prod2*(1.0-prod3)

        intra_comm=0.0
        # for comm in tmp_comm:
        #     intra_comm+=self.inf_community(comm,nodes_in_new_comm)
        #     intra_comm +=0.01*len(self.comm2node[comm])
        for extra_node in more_nodes:
            # if self.comm[extra_node] in tmp_comm:
            X=np.array([self.comm_deg_centra[self.comm[extra_node]][extra_node],
                        self.comm_between_centra[self.comm[extra_node]][extra_node],
                        self.comm_load_centra[self.comm[extra_node]][extra_node],
                        self.comm_avg_nei_deg[self.comm[extra_node]][extra_node],
                        self.comm_harmonic_centra[self.comm[extra_node]][extra_node],
                        self.comm_close_centra[self.comm[extra_node]][extra_node]]).reshape(1,-1)
            intra_comm+=float(self.classifiers[self.comm[extra_node]].predict(X)[0])#+self.exscore[extra_node]

        # X=np.array([self.deg_centra[node],self.between_centra[node],self.load_centra[node],self.avg_neigh_deg[node]]).reshape(1,-1)
        # print inf2-inf1,intra_comm
        return (inf2-inf1)+intra_comm*0.01#+self.exscore[node]*0.001
        # return self.graph.out_degree(node)+nx.average_neighbor_degree(self.graph,nodes=[node])[node]+self.exscore[node]

    def add_seed(self,seed_set,neigh_node,neigh_comm,node,total_cost):
        for item in self.graph.successors(node):
            comm=self.comm[item]
            neigh_node.add(item)
            neigh_comm.add(comm)
        seed_set.append(node)
        total_cost+=self.struc_cost(node,3)
        return seed_set,neigh_node,neigh_comm,total_cost

    def node_expansion(self,k,gamma):
        print "Finding top",k,"nodes"
        print "No.\tNode_id\tRemaining_balance"
        # tolscore={}
        # for node in self.graph.nodes():
        #     X = np.array([self.comm_deg_centra[self.comm[node]][node],
        #                   self.comm_between_centra[self.comm[node]][node],
        #                   self.comm_load_centra[self.comm[node]][node],
        #                   self.comm_avg_nei_deg[self.comm[node]][node],
        #                   self.comm_harmonic_centra[self.comm[node]][node],
        #                   self.comm_close_centra[self.comm[node]][node]]).reshape(1, -1)
        #     pred = float(self.classifiers[self.comm[node]].predict(X)[0])
        #     tolscore[node]=pred+self.exscore[node]
        # print score
        avg_ext=0.0
        # avg_nei_deg=nx.average_neighbor_degree(self.graph)
        for node in self.graph.nodes():
            # print self.exscore[node]
            avg_ext+=self.exscore[node]
        avg_ext = float(avg_ext/self.graph.number_of_nodes())
        # print avg_ext
        pairs=dict()
        for node in self.graph.nodes():
            # print score[node]
            if self.exscore[node]<avg_ext:
                continue
            score = self.get_score(node,gamma)
            tmp=sorted(pairs.items(),key=lambda item:item[1],reverse=False)
            if len(pairs)>=10*k and score<=tmp[0][1]:
                continue
            pairs[node]=score
            if len(pairs)>10*k:
                pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=False)[1:])
        # pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True))
        updated=list()
        for i in xrange(0,self.graph.number_of_nodes()):
            updated.append(True)
        seed_set=[]
        neigh_node=set()
        neigh_comm=set()
        # total_score=0.0
        total_cost=0.0
        start_time=time.time()
        for i in xrange(0,k):
            best_pair = sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
            pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            while(updated[int(best_pair[0])] is not True):
                m_gain=self.marginal_gain(seed_set,neigh_node,neigh_comm,best_pair[0],gamma,avg_ext)
                updated[best_pair[0]]=True
                pairs[best_pair[0]]=m_gain
                best_pair=sorted(pairs.items(),key=lambda item:item[1],reverse=True)[0]
                pairs=dict(sorted(pairs.items(),key=lambda item:item[1],reverse=True)[1:])
            seed_set,neigh_node,neigh_comm,total_cost=self.add_seed(seed_set,neigh_node,neigh_comm,best_pair[0],total_cost)
            if total_cost>=self.budget:
                break
            # total_score+=best_pair[1]
            print i+1,"\t",best_pair[0],"\t",self.budget-total_cost
            for i in xrange(0,self.graph.number_of_nodes()):
                updated[i]=False
            # print(len(seed_set))
        print "Seed selection:",time.time()-start_time,"s"
        return seed_set


def write_seeds(seeds):
    with open('seeds-1.3.txt','w') as f:
        for seed in list(seeds):
            f.write(str(seed))
            f.write('\n')

def load_seeds(seeds_path):
    seeds=[]
    with open(seeds_path,'r') as f:
        for line in f:
            seeds.append(int(line.strip()))
    return seeds



    # def draw_graph(self):
    #     pos=nx.spring_layout(self.graph)
    #     edge_labels=dict([((u,v,),d['weight']) for u,v,d in self.graph.edges(data=True)])
    #     nx.draw_networkx_edge_labels(self.graph,pos,edge_labels=edge_labels)
    #     nx.draw(self.graph,pos=pos,node_size=100,arrows=True)
    #     plt.show()

if __name__ == '__main__':
    print platform.uname()[0],platform.uname()[1],platform.processor(),virtual_memory().total/(1024**3)
    network='../weighted_directed_nets/network.dat'
    # network='../com-dblp.ungraph.txt/com-dblp.ungraph.txt'
    # network='../email-eu/email-Eu-core.txt'
    community='../weighted_directed_nets/community.dat'
    # community='../com-dblp.all.cmty.txt/com-dblp.all.cmty.txt'
    # community='../email-eu/email-Eu-core-department-labels.txt'
    # network='NetHEHT.txt'
    # community='NetHEHT_com.txt'
    budget=20000.0
    excofim = extended_CoFIM(network,community,0,budget,directed=True,weighted=True,parallel = True)
    # l=300
    # for comm in excofim.comm2node.keys():
    #    if l>len(excofim.comm2node[comm]):
    #        l=len(excofim.comm2node[comm])
    # print l
    seedNo=50
    seeds=excofim.node_expansion(seedNo,3)
    # formerseeds=load_seeds("seeds-1.txt")
    # write_seeds(seeds)
    # a=0
    # for seed in seeds:
    #     if seed not in formerseeds:
    #        / print "Not same!"
    #         break
    #     else:
    #         a+=1
    # if a==50:
    #     print "same!"
    # print seeds
    # inf=Evaluation.monte_carlo_extend(excofim,seeds,k=50,num_simu=100)
    # print inf
    seed=[50]
    for num in seed:
        inf=Evaluation.mc_method(excofim.graph,seeds,k=num,num_simu=100)
        print "Seeds number:",num,"Total influence:",inf
    # excofim.draw_graph()
