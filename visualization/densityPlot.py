
import os
import numpy as np
import pandas as pd
import random
import math
import warnings # to create a warning
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Principal component analysis
from sklearn.ensemble import RandomForestClassifier # Random Forest classifier
from sklearn.metrics import confusion_matrix # To compute Confusion Matrix
from sklearn.cross_validation import StratifiedShuffleSplit # Train/Test split
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#import tsne # t-Distributed Stochastic Neighbor Embedding (t-SNE) 


posfilename = '../data/node_values.csv'


valsDF = pd.read_csv(posfilename) 
valsDF=valsDF[valsDF['dist_nn']<5]

#valsDF=valsDF[0:200]
N = len(valsDF)

NFeats = 6  # number of features of the herd (mean nn, variance in nn, mean of 3 neighbors, circular variance)
featRaw = np.empty((N,NFeats)) # 250 images x 4096 pixel values - raw features


distances = np.zeros((N,N))
xvals = valsDF['x'].values
yvals = valsDF['y'].values
dist_nn = valsDF['dist_nn'].values
angle_nn = valsDF['angle_nn'].values
align_nn = valsDF['align_nn'].values


for i in range(N):
    for j in range(i+1,N):
        distances[i,j]=math.sqrt((xvals[i]-xvals[j])**2+(yvals[i]-yvals[j])**2)
        distances[j,i]=distances[i,j]

length = 10
weights = np.exp(-np.power(distances/length,2))
sumweights=np.sum(weights,axis=1)
for i in range(N):
    featRaw[i][0]=np.dot(weights[i],dist_nn)/sumweights[i]
    featRaw[i][1]=np.dot(weights[i],angle_nn)/sumweights[i]
    featRaw[i][2]=np.dot(weights[i],align_nn)/sumweights[i]
    featRaw[i][3]=np.dot(weights[i],np.power(dist_nn-featRaw[i][0],2))/sumweights[i]
    featRaw[i][4]=np.dot(weights[i],np.power(angle_nn-featRaw[i][1],2))/sumweights[i]
    featRaw[i][5]=np.dot(weights[i],np.power(align_nn-featRaw[i][2],2))/sumweights[i]
    
# # Visualisation
# 
# Now that we have a set of features (Naive and HOGs) we will use linear (PCA) and non-linear (t-SNE) dimensionality reduction techniques to visualise the data and see whether we can detect some structure in the data.

featRaw2=featRaw[::2,:]

#-----------------------------------------------------------------------------#
# Visualise high-dimensional data
#-----------------------------------------------------------------------------#
k = 2 # 2D projection
seed=20 # to reproduce t-SNE results
# PCA
#pcaOut = PCA(n_components=k)
#pcaOut.fit(featRaw.T)
#embedding = pcaOut.components_.T
#plot2D(embedding, classLabel, "PCA Raw Features", "lower left")
# t-SNE
model = TSNE(n_components=3, random_state=0)

embedding = model.fit_transform(featRaw) 
plt.plot(embedding[:,0],embedding[:,1],'.')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:,0],embedding[:,1],embedding[:,2],'.')
#embedding = tsne.tsne(featRaw, no_dims=k, initial_dims=20, perplexity=10.0, seed=seed)
