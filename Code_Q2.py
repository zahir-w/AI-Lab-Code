import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl

DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 2), (1, 3, 3), (1, 5, 1), (1, 8, 7), (2, 3, 3), (3, 4, 1), (4, 8, 2),
                            (5, 7, 2), (6, 12, 1), (6, 13, 2), (7, 6, 1), (7, 8, 5), (8, 9, 2), (8, 14, 3),
                            (11, 8, 4), (14, 10, 1),(14, 11, 2)])
ans = []
for target in DG.nodes():
    length = nx.dijkstra_path_length(DG, 1, target, weight='weight')
    ans.append((target, length))
ans.sort(key=lambda x: x[0])
print(ans)

f = open("Variables_Q2.pkl", "wb")
pkl.dump({"Q2": ans}, f)
f.close()