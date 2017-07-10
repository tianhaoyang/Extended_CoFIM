#coding:utf8
import networkx as nx
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import Evaluation

# def mc_method(graph,seed):
# 	num_simu=100
# 	random.seed()
# 	inf=0.0
# 	for r in xrange(0,num_simu):
# 		A=list(seed)
# 		visited={}
# 		for node in graph.nodes():
# 			visited[node]=False
# 		for v in A:
# 			if visited[v] is False:
# 				for neighbour in graph.successors(v):
# 					pp=graph[v][neighbour]['weight']/sum(graph[p][neighbour]['weight'] for p in graph.predecessors(neighbour))
# 					rand=random.random()
# 					if rand<=pp:
# 						A.append(neighbour)
# 				visited[v]=True
# 		inf+=len(A)
# 	return float(inf/num_simu)

def training(community):
	# print "Training classifiers..."
	sampleSize=200
	if nx.number_of_nodes(community)>sampleSize:
		print "Network sampling..."
		current = random.sample(community.nodes(),1)[0]
		sampleNodes = [current]
		graph=community.subgraph(sampleNodes)
		while(nx.number_of_nodes(graph)<sampleSize):
			w=random.sample(nx.edges(community,current),1)
			nxt=w[0][1]
			p = random.random()
			threshold = float(nx.degree(community,current))/float(nx.degree(community,nxt))
			if p<threshold:
				if nxt not in sampleNodes:
					sampleNodes.append(nxt)
				current = nxt
			graph=community.subgraph(sampleNodes)
	else:
		graph=community

	degree=graph.out_degree(graph.nodes())
	ordered=sorted(degree.items(),key=lambda item:item[1],reverse=True)

	numBins=20
	binSize = int(nx.number_of_nodes(graph)/numBins)

	bins = dict()
	values = []
	key = 0
	# print ordered

	for index, i in enumerate(ordered):
		# if adding to current bin is greater than bin size
		if len(values)+1>binSize:
			# if the metric value of the current node is not the same as the last node
			# then increment to next bin
			if ordered[index-1][1] != i[1]:
				key = key+1
				values = []
		# append the node ID
		values.append(i[0])
		bins[key]=values

	numSeeds=20
	final_seeds=set()
	for binNo in bins.keys():
		# final_seeds.add(random.sample(bins[binNo],1)[0])
		possibleSeeds=bins[binNo]
		if len(possibleSeeds)>=numSeeds:
			seeds=random.sample(possibleSeeds,numSeeds)
		else:
			seeds=possibleSeeds
			if binNo+1 in bins.keys():
				remainder=numSeeds-len(possibleSeeds)
				if remainder>len(bins[binNo+1]):
					s=bins[binNo+1]
				else:
					s=random.sample(bins[binNo+1],remainder)
				seeds.extend(s)
		# print seeds
		final_seeds.add(random.sample(seeds,1)[0])
	# graph=community
	# final_seeds=community.nodes()

	sample_per_seed=1
	if len(graph)==1:
		# print("yes")
		deg_centra={graph.nodes()[0]:1}
	else:
		deg_centra=nx.degree_centrality(graph)
	between_centra=nx.betweenness_centrality(graph)
	load_centra=nx.load_centrality(graph)
	# eigen_centra2=nx.eigenvector_centrality(graph)
	avg_neigh_deg=nx.average_neighbor_degree(graph)
	harmonic_centra=nx.harmonic_centrality(graph)
	close_centra=nx.closeness_centrality(graph)
	feature=np.zeros([len(final_seeds)*sample_per_seed,6])
	influence=[]
	for i,seed in enumerate(list(final_seeds)):
		for n in xrange(0,sample_per_seed):
			inf=Evaluation.mc_method(graph,[seed],k=1,num_simu = 100)
			feature[i*sample_per_seed+n,0]=deg_centra[seed]
			feature[i*sample_per_seed+n,1]=between_centra[seed]
			feature[i*sample_per_seed+n,2]=load_centra[seed]
			# feature[i*20+n,3]=eigen_centra[seed]
			feature[i*sample_per_seed+n,3]=avg_neigh_deg[seed]
			feature[i*sample_per_seed+n,4]=harmonic_centra[seed]
			feature[i*sample_per_seed+n,5]=close_centra[seed]
			influence.append(inf)
			# print feature[n,],inf


	model=LinearRegression(normalize=True)
	classifier=model.fit(feature,influence)

	return classifier


# if __name__=='__main__':
# 	G = nx.DiGraph()
# 	with open('../weighted_directed_nets/network3.dat','r') as f:
# 		for i, line in enumerate(f):
# 			node1,node2,weight=line.strip().split('\t')
# 			G.add_edge(int(node1)-1,int(node2)-1,weight=float(weight))
# 	training(G)