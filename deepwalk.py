import pandas as pd
skill_data = pd.read_csv('Technology Skills.csv')
skill_data.head()

import networkx as nx
edges = skill_data[['Title','Example']].values.tolist()
net = nx.from_edgelist(edges)

def random_walk(graph,seed,rounds=20):
  import random
  movements = [seed]
  for round in range(rounds):
    node_choices = [node for node in graph.neighbors(seed)]
    seed = random.choice(node_choices)
    movements.append(seed)
  return movements

random_walk(net,'Python')

walks = []
vertices = [n for n in net.nodes]
for v in vertices:
  walks.append(random_walk(graph=net,seed=v))
  
from gensim.models.word2vec import Word2Vec
embeddings = Word2Vec(walks,size=10,window=5)

# embeddings.most_similar('C++') ## verify results are sensible

embeddings.save("graph2vec2.model")
array_dict = {node:embeddings[node] for node in net.nodes if node in embeddings}
embedded_nodes = [node for node in net.nodes if node in array_dict]
import numpy as np
arrays = np.array([array_dict[node] for node in embedded_nodes])

skills = [skill for skill in skill_data['Example'].unique()]
jobs = [job for job in skill_data['Title'].unique()]
skill_idx = [idx for idx,elem in enumerate(embedded_nodes) if elem in skills]
job_idx = [idx for idx,elem in enumerate(embedded_nodes) if elem in jobs]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
p_comps = pca.fit_transform(arrays)

import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim,ylim

# xlim(-13,13)
# ylim(-13,13)
plt.scatter(
    # Jobs are red, skills are blue
    x=p_comps[:,0],y=p_comps[:,1],color=['b' if idx in skill_idx else 'r' for idx in range(len(arrays))],
    marker='+',
    alpha = 0.35,
    )
    

