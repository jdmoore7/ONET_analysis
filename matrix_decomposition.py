import pandas as pd
skill_data = pd.read_csv('Technology Skills.csv')
skill_data.head()

x = pd.get_dummies(skill_data.set_index('Title')['Example'])

x = x.groupby(lambda var:var, axis=0).sum()

cols = x.columns.to_list()
rows = x.transpose().columns.to_list()

y = x.to_numpy()

import torch 
import torch.nn as nn
import torch.nn.functional as F

job_skill_tensor = torch.FloatTensor(y)


import random
class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_jobs=len(rows), n_skills=len(cols), n_factors=10):
        super().__init__()
        
        self.job_latent = nn.Parameter(torch.rand(n_jobs,n_factors))
        self.skill_latent = nn.Parameter(torch.rand(n_factors, n_skills))
        
        
    def forward(self):
        return torch.mm(self.job_latent,self.skill_latent)


model = MatrixFactorization()
loss_fn = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
epochs = 1000
for epoch in range(epochs):
    loss = 0
    
    prediction = model.forward()
    loss += loss_fn(prediction, job_skill_tensor)
    losses.append(loss)

    # Reset the gradients to 0
    optimizer.zero_grad()

    # backpropagate
    loss.backward()

    # update weights
    optimizer.step()
    if epoch % 50 == 0:
        print(loss)
        
from sklearn.metrics.pairwise import cosine_similarity
job_features = np.array(model.job_latent.detach())
skill_features = np.array(model.skill_latent.detach())
job_skill_stacked = np.concatenate((job_features,skill_features.transpose()))
job_skill_sim = cosine_similarity(job_skill_stacked)

import operator
entities = []
entities.extend(rows + cols)

def get_similar(node,sim_threshold=None,count_threshold=None,category=None):
  idx = entities.index(node)
  sim_scores = job_skill_sim[idx]
  retrieved = [(elem,score) for elem,score in zip(entities,sim_scores)]

  if category == 'jobs':
    retrieved = [tup for idx,tup in enumerate(retrieved) if idx < len(rows)]
  elif category == 'skills':
    retrieved = [tup for idx,tup in enumerate(retrieved) if idx > len(rows)]
  else:
    pass
  
  
  if sim_threshold:
    retrieved = [(elem,score) for elem,score in retrieved if score > sim_threshold]
  
  retrieved = sorted(retrieved,key=operator.itemgetter(1),reverse=True)

  if count_threshold:
    retrieved = [tup for idx,tup in enumerate(retrieved) if idx < count_threshold]  
  
  return retrieved

get_similar('Python',category='jobs',sim_threshold=0.8,count_threshold=25)

# Save latent feature similarity values in a pickled file!

import pickle
with open('cos_sim_pickle.pkl', 'wb') as f:
  pickle.dump(job_skill_sim, f)

with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)  

with open('latent_features.pkl', 'wb') as f:
  pickle.dump(job_skill_stacked,f)    
  
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
p_comps = pca.fit_transform(job_skill_stacked)

import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim,ylim

plt.scatter(
    x=p_comps[:,0],y=p_comps[:,1],color=['r' if idx < len(rows) else 'b' for idx in range(job_skill_stacked.shape[0])],
    marker='+',
    alpha = 0.25,
    )
    
    

