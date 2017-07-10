#coding:utf8
import community
import networkx as nx
import matplotlib.pyplot as plt

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
#G = nx.erdos_renyi_graph(30, 0.05)
G = nx.Graph()
with open('NetHEHT.txt','r') as f:
    for i, line in enumerate(f):
        if i == 0:
            num_node, num_edge = line.strip().split('\t')
            continue
        node1,node2=line.strip().split('\t')
        G.add_edge(int(node1),int(node2))
#first compute the best partition
partition = community.best_partition(G)

#drawing
size = float(len(set(partition.values())))
# pos = nx.spring_layout(G)
# count = 0.
with open('NetHEHT_com2.txt','w') as fr:
    for com in set(partition.values()) :
        # count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]

        for node in list_nodes:
            fr.write(str(node))
            fr.write(' ')
        fr.write('\n')



# nx.draw_networkx_edges(G,pos, alpha=0.5)
# plt.show()
