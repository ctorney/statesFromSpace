

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Principal component analysis
from sklearn.ensemble import RandomForestClassifier # Random Forest classifier
from sklearn.metrics import confusion_matrix # To compute Confusion Matrix
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#import tsne # t-Distributed Stochastic Neighbor Embedding (t-SNE) 


posfilename = '../data/node_values.csv'


valsDF = pd.read_csv(posfilename) 
#valsDF=valsDF[valsDF['dist_nn']<25]

#valsDF=valsDF[0:200]
N = len(valsDF)

NFeats = 2 # number of features of the herd (mean nn, variance in nn, mean of 3 neighbors, circular variance)
featRaw = np.empty((N,NFeats)) # 250 images x 4096 pixel values - raw features


distances = np.zeros((N,N))
xvals = valsDF['x'].values
yvals = valsDF['y'].values
#dist_nn = valsDF['dist_nn'].values
angle_nn = valsDF['angle_knn'].values
align_nn = valsDF['align_knn'].values
angles = valsDF['angle'].values
c2angle = np.cos(2*angles)
s2angle = np.sin(2*angles)

#for i in range(N):
 #   for j in range(i+1,N):
  #      distances[i,j]=math.sqrt((xvals[i]-xvals[j])**2+(yvals[i]-yvals[j])**2)
   #     distances[j,i]=distances[i,j]

#length = 10
#weights = np.exp(-np.power(distances/length,2))


length = 30
weights = np.zeros_like(distances)
weights[distances<length]=1.0
sumweights=np.sum(weights,axis=1)
featRaw[:,0]=angle_nn
featRaw[:,1]=align_nn
#for i in range(N):
 #   j=0
    #av_dist_nn=np.dot(weights[i],dist_nn)/sumweights[i]
    #featRaw[i][j]=av_dist_nn
    #j+=1
    
    #featRaw[i][j]=np.dot(weights[i],angle_nn)/sumweights[i]
    #j+=1
    
    #featRaw[i][j]=np.dot(weights[i],align_nn)/sumweights[i]
    #j+=1
    
    #featRaw[i][j]=np.dot(weights[i],np.power(dist_nn-av_dist_nn,2))/sumweights[i]
  #  j+=1
    
    #featRaw[i][j]=np.dot(weights[i],np.power(angle_nn-featRaw[i][1],2))/sumweights[i]
    #j+=1
    
    #featRaw[i][j]=np.dot(weights[i],np.power(align_nn-featRaw[i][2],2))/sumweights[i]
    #j+=1
    
   # sinav = np.dot(weights[i],s2angle)/sumweights[i]
   # cosav = np.dot(weights[i],c2angle)/sumweights[i]
   # featRaw[i][j]=(sinav**2+cosav**2)**0.5
    
    #j+=1
    #featRaw[i][j]=np.sum(weights[i])
    
# # Visualisation
# 
# Now that we have a set of features (Naive and HOGs) we will use linear (PCA) and non-linear (t-SNE) dimensionality reduction techniques to visualise the data and see whether we can detect some structure in the data.



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from sklearn.metrics import calinski_harabaz_score

nk=4
kmm = KMeans(n_clusters=nk)

kmm.fit(featRaw)

labels = kmm.labels_

silhouette_score(featRaw,labels)
#labels[ps[:,0]<sw]=0
#labels[ps[:,0]>=sw]=1

plt.figure()
plt.scatter(featRaw[:,0],featRaw[:,1], c=labels.astype(np.float))



gapstats = gap.gap(featRaw)

featRaw2=featRaw[::2,:]

#-----------------------------------------------------------------------------#
# Visualise high-dimensional data
#-----------------------------------------------------------------------------#
k = 2 # 2D projection
seed=20 # to reproduce t-SNE results
# PCA
pcaOut = PCA(n_components=k)
pcaOut.fit(featRaw.T)
embedding = pcaOut.components_.T
#plot2D(embedding, classLabel, "PCA Raw Features", "lower left")
# t-SNE
model = TSNE(n_components=2, random_state=0,perplexity=5)

embedding = model.fit_transform(featRaw) 
plt.plot(embedding[:,0],embedding[:,1],'.')
plt.figure()
plt.hist2d(embedding[:,0],embedding[:,1], bins=40)
#plt.colorbar()
#plt.show()
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(featRaw[:,0],featRaw[:,1],featRaw[:,2],s=0.1)
#embedding = tsne.tsne(featRaw, no_dims=k, initial_dims=20, perplexity=10.0, seed=seed)
