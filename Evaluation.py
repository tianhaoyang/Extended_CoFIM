#coding:utf8
import random

S_STATE=0
I_STATE=1
SI_STATE=2
R_STATE=3

def monte_carlo(graph,seed,k,num_simu=10000):
	random.seed()
	inf=0.0
	i_size=0
	r_size=0
	si_size=0
	i_arr=[0]*graph.num_node
	r_arr=[0]*graph.num_node
	si_arr=[0]*graph.num_node
	state=[S_STATE]*graph.num_node
	for r in xrange(0,num_simu):
		active_size=0
		for i in xrange(0,r_size):
			state[r_arr[i]]=S_STATE
		r_size=0

		for i in xrange(0,k):
			i_arr[i_size]=seed[i]
			state[i_arr[i]]=I_STATE
			i_size+=1

		while(i_size>0):
			active_size+=i_size
			si_size=0
			for i in xrange(0,i_size):
				for node in graph.graph.neighbors(i_arr[i]):
					if state[node]==S_STATE:
						pp=1.0/graph.graph.degree(node)
						rand=random.random()
						if rand<pp:
							state[node]=SI_STATE
							si_arr[si_size]=node
							si_size+=1
			for i in xrange(0,i_size):
				state[i_arr[i]]=R_STATE
				r_arr[r_size]=i_arr[i]
				r_size+=1
			i_size=0
			for i in xrange(0,si_size):
				state[si_arr[i]]=I_STATE
				i_arr[i_size]=si_arr[i]
				i_size+=1
		inf+=active_size
	#print "Total influence:",inf/num_simu
	return inf/num_simu

def monte_carlo_extend(graph,seed,k=None,num_simu=10000):
	random.seed()
	inf=0.0
	i_size=0
	r_size=0
	si_size=0
	i_arr=[0]*graph.num_node
	r_arr=[0]*graph.num_node
	si_arr=[0]*graph.num_node
	state=[S_STATE]*graph.num_node
	if k==None:
		k=len(seed)
	for r in xrange(0,num_simu):
		active_size=0
		for i in xrange(0,r_size):
			state[r_arr[i]]=S_STATE
		r_size=0

		for i in xrange(0,k):
			i_arr[i_size]=seed[i]
			state[i_arr[i]]=I_STATE
			i_size+=1

		while(i_size>0):
			active_size+=i_size
			si_size=0
			for i in xrange(0,i_size):
				for node in graph.graph.successors(i_arr[i]):
					if state[node]==S_STATE:
						# pp=1.0/graph.graph.in_degree(node)
						pp=graph.graph[i_arr[i]][node]['weight']/sum(graph.graph[pre][node]['weight'] for pre in graph.graph.predecessors(node))
						# if graph.comm[node]==graph.comm[i_arr[i]]:
						# 	pp*=(1.0+0.3)
						# else:
						# 	pp*=(1.0-0.3)
						rand=random.random()
						if rand<=pp:
							state[node]=SI_STATE
							si_arr[si_size]=node
							si_size+=1
				# a=random.choice([0])
				# max_weight=max(graph.graph[i_arr[i]][suc]['weight'] for suc in graph.graph.successors(i_arr[i]))
				# max_out=max(graph.graph.out_degree(suc) for suc in graph.graph.successors(i_arr[i]))
				# for j in xrange(0,a):
				# 	for node in graph.graph.successors(i_arr[i]):
				# 		if state[node]==S_STATE:
				# 			if graph.graph[i_arr[i]][node]['weight']>0.8*max_weight and graph.graph.out_degree(node)>0.8*max_out:
				# 				pp=graph.graph[i_arr[i]][node]['weight']/sum(graph.graph[pre][node]['weight'] for pre in graph.graph.predecessors(node))
				# 				rand=random.random()
				# 				if rand<pp:
				# 					state[node]=SI_STATE
				# 					si_arr[si_size]=node
				# 					si_size+=1
			for i in xrange(0,i_size):
				state[i_arr[i]]=R_STATE
				r_arr[r_size]=i_arr[i]
				r_size+=1
			i_size=0
			for i in xrange(0,si_size):
				state[si_arr[i]]=I_STATE
				i_arr[i_size]=si_arr[i]
				i_size+=1
		inf+=active_size
	#print "Total influence:",inf/num_simu
	return inf/num_simu

def monte_carlo_extend2(graph,seed,k=None,num_simu=10000):
	random.seed()
	inf=0.0
	i_size=0
	r_size=0
	si_size=0
	i_arr=[0]*graph.num_node
	r_arr=[0]*graph.num_node
	si_arr=[0]*graph.num_node
	state=[S_STATE]*graph.num_node
	if k==None:
		k=len(seed)
	for r in xrange(0,num_simu):
		active_size=0
		for i in xrange(0,r_size):
			state[r_arr[i]]=S_STATE
		r_size=0

		for i in xrange(0,k):
			i_arr[i_size]=seed[i]
			state[i_arr[i]]=I_STATE
			i_size+=1

		while(i_size>0):
			active_size+=i_size
			si_size=0
			for i in xrange(0,i_size):
				for node in graph.graph.successors(i_arr[i]):
					if state[node]==S_STATE:
						# pp=1.0/graph.graph.in_degree(node)
						pp=graph.graph[i_arr[i]][node]['weight']/sum(graph.graph[pre][node]['weight'] for pre in graph.graph.predecessors(node))
						# if graph.comm[node]==graph.comm[i_arr[i]]:
						# 	pp*=(1.0+0.3)
						# else:
						# 	pp*=(1.0-0.3)
						rand=random.random()
						if rand<=pp:
							state[node]=SI_STATE
							si_arr[si_size]=node
							si_size+=1
				a=random.choice([1])
				max_weight=max(graph.graph[i_arr[i]][suc]['weight'] for suc in graph.graph.successors(i_arr[i]))
				max_out=max(graph.graph.out_degree(suc) for suc in graph.graph.successors(i_arr[i]))
				for j in xrange(0,a):
					for node in graph.graph.successors(i_arr[i]):
						if state[node]==S_STATE:
							if graph.graph[i_arr[i]][node]['weight']>0.5*max_weight and graph.graph.out_degree(node)>0.9*max_out:
								pp=graph.graph[i_arr[i]][node]['weight']/sum(graph.graph[pre][node]['weight'] for pre in graph.graph.predecessors(node))
								rand=random.random()
								if rand<pp:
									state[node]=SI_STATE
									si_arr[si_size]=node
									si_size+=1
			for i in xrange(0,i_size):
				state[i_arr[i]]=R_STATE
				r_arr[r_size]=i_arr[i]
				r_size+=1
			i_size=0
			for i in xrange(0,si_size):
				state[si_arr[i]]=I_STATE
				i_arr[i_size]=si_arr[i]
				i_size+=1
		inf+=active_size
	#print "Total influence:",inf/num_simu
	return inf/num_simu


def monte_carlo_extend3(graph,seed,k=None,num_simu=10000):
	random.seed()
	inf=0.0
	i_size=0
	r_size=0
	si_size=0
	i_arr=[0]*graph.number_of_nodes()
	r_arr=[0]*graph.number_of_nodes()
	si_arr=[0]*graph.number_of_nodes()
	state=[S_STATE]*graph.number_of_nodes()
	if k==None:
		k=len(seed)
	for r in xrange(0,num_simu):
		active_size=0
		for i in xrange(0,r_size):
			state[r_arr[i]]=S_STATE
		r_size=0

		for i in xrange(0,k):
			i_arr[i_size]=seed[i]
			state[i_arr[i]]=I_STATE
			i_size+=1

		while(i_size>0):
			active_size+=i_size
			si_size=0
			for i in xrange(0,i_size):
				for node in graph.successors(i_arr[i]):
					if state[node]==S_STATE:
						# pp=1.0/graph.graph.in_degree(node)
						pp=graph[i_arr[i]][node]['weight']/sum(graph[i_arr[i]][suc]['weight'] for suc in graph.successors(i_arr[i]))
						# if graph.comm[node]==graph.comm[i_arr[i]]:
						# 	pp*=(1.0+0.3)
						# else:
						# 	pp*=(1.0-0.3)
						rand=random.random()
						if rand<=pp:
							state[node]=SI_STATE
							si_arr[si_size]=node
							si_size+=1
				# a=random.choice([0])
				# max_weight=max(graph.graph[i_arr[i]][suc]['weight'] for suc in graph.graph.successors(i_arr[i]))
				# max_out=max(graph.graph.out_degree(suc) for suc in graph.graph.successors(i_arr[i]))
				# for j in xrange(0,a):
				# 	for node in graph.graph.successors(i_arr[i]):
				# 		if state[node]==S_STATE:
				# 			if graph.graph[i_arr[i]][node]['weight']>0.8*max_weight and graph.graph.out_degree(node)>0.8*max_out:
				# 				pp=graph.graph[i_arr[i]][node]['weight']/sum(graph.graph[pre][node]['weight'] for pre in graph.graph.predecessors(node))
				# 				rand=random.random()
				# 				if rand<pp:
				# 					state[node]=SI_STATE
				# 					si_arr[si_size]=node
				# 					si_size+=1
			for i in xrange(0,i_size):
				state[i_arr[i]]=R_STATE
				r_arr[r_size]=i_arr[i]
				r_size+=1
			i_size=0
			for i in xrange(0,si_size):
				state[si_arr[i]]=I_STATE
				i_arr[i_size]=si_arr[i]
				i_size+=1
		inf+=active_size
	#print "Total influence:",inf/num_simu
	return inf/num_simu

def mc_method(graph,seed,k=None,num_simu=10000):
	inf=0.0
	random.seed()
	tmp_seed=[]
	for i in xrange(0,k):
		tmp_seed.append(seed[i])

	for r in xrange(0,num_simu):
		A=list(tmp_seed)
		visited={}
		for node in graph.nodes():
			visited[node]=False
		for v in A:
			if visited[v] is False:
				for neighbour in graph.neighbors(v):
					pp=graph[v][neighbour]['weight']/sum(graph[p][neighbour]['weight'] for p in graph.predecessors(neighbour))
					rand=random.random()
					if rand<=pp:
						A.append(neighbour)
				visited[v]=True
		inf+=len(A)
	return inf/num_simu

def mc_method1(graph,seed,k=None,num_simu=10000):
	inf=0.0

	for r in xrange(0,num_simu):
		A=list(seed)
		visited=[False]*graph.num_node
		for v in A:
			if visited[v] is False:
				for neighbour in graph.graph.neighbors(v):
					pp=1.0/graph.graph.degree(neighbour)
					rand=random.random()
					if rand<=pp:
						A.append(neighbour)
				visited[v]=True
		inf+=len(A)
	return inf/num_simu

# def mc_method2(graph,subgraph,seed,k=None,num_simu=10000):
# 	inf=0.0
#
# 	for r in xrange(0,num_simu):
# 		A=list(seed)
# 		visited=[False]*graph.number_of_nodes()
# 		for v in A:
# 			if visited[v] is False:
# 				for neighbour in graph.neighbors(v):
# 					if neighbour in subgraph.nodes():
# 						pp=graph[v][neighbour]['weight']/sum(graph[v][suc]['weight'] for suc in graph.successors(v))
# 						rand=random.random()
# 						if rand<=pp:
# 							A.append(neighbour)
# 				visited[v]=True
# 		inf+=len(A)
# 	return inf/num_simu