import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl

G = nx.Graph()
G.add_nodes_from(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V'])
G.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('A', 'G'), ('B', 'E'), ('B', 'I'), ('C', 'D'), ('C', 'F'),
     ('D', 'F'), ('D', 'G'), ('E', 'G'), ('E', 'H'), ('E', 'I'), ('F', 'G'), ('F', 'K'), ('G', 'H'), ('G', 'K'),
     ('G', 'L'), ('G', 'N'), ('H', 'J'), ('H', 'L'), ('I', 'Q'), ('I', 'V'), ('K', 'N'), ('K', 'R'), ('L', 'N'),
     ('L', 'O'), ('L', 'Q'), ('M', 'N'), ('M', 'O'), ('M', 'P'), ('M', 'R'), ('M', 'S'), ('N', 'R'), ('O', 'Q'),
     ('O', 'U'), ('P', 'R'), ('P', 'S'), ('Q', 'U'), ('S', 'T'), ('S', 'U'), ('T', 'U'), ('T', 'V'), ('U', 'V')])
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()
T1 = nx.dfs_tree(G, 'A')
print(T1.edges())
T2 = nx.bfs_tree(G, 'A')
print(T2.edges())

ans3 = []
for target in G.nodes():
    path = nx.dijkstra_path(G, 'A', target, weight='weight')
    ans3.append((target, len(path)-1))
print(ans3)

answer = {'Q1_i': G, 'Q1_ii_DFS': T1.edges(), 'Q1_ii_BFS': T2.edges(), 'Q1_iii': ans3}
f = open('Variables_Q1.pkl', 'wb')
pkl.dump(answer, f)
f.close()
