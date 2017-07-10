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
				for j in xrange(0,1):
					for node in graph.graph.successors(i_arr[i]):
						if state[node]==S_STATE:
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
