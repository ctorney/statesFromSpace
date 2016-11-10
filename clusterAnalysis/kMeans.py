
import os
import numpy as np
import pandas as pd
import random
import math
import warnings # to create a warning
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Principal component analysis
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import mixture
#import tsne # t-Distributed Stochastic Neighbor Embedding (t-SNE) 
import itertools
from scipy import linalg

import matplotlib as mpl

posfilename = '../data/nodes_meters.csv'

posDF = pd.read_csv(posfilename) 

N = len(posDF)


NFeats = 2  # number of features of the herd (mean nn, variance in nn, mean of 3 neighbors, circular variance)
featRaw = np.empty((N,NFeats)) # 250 images x 4096 pixel values - raw features


distances = np.zeros((N,N))
distancesKD = -np.ones((N,N))

angles = posDF['angle'].values
c2angle = np.cos(2*angles)
s2angle = np.sin(2*angles)
c2angle[np.isnan(angles)]=0
s2angle[np.isnan(angles)]=0

xvals=posDF['xm'].values
yvals=posDF['ym'].values

for i in range(N):
    for j in range(i+1,N):
        distances[i,j]=math.sqrt((xvals[i]-xvals[j])**2+(yvals[i]-yvals[j])**2)
        distances[j,i]=distances[i,j]
length = 50
weights = np.zeros_like(distances)
weights[distances<length]=1.0
sumweights=np.sum(weights,axis=1)


for i in range(N):
    for j in range(1,N):
        if math.isfinite(angles[j]):
            
            distancesKD[i,j]=distances[i,j]
        

weightsKD = np.zeros_like(distances)
weightsKD[distancesKD<length]=1.0
weightsKD[distancesKD<0]=0.0
sumweightsKD=np.sum(weightsKD,axis=1)

for i in range(N):
    j=0
    #av_dist_nn=np.dot(weights[i],dist_nn)/sumweights[i]
    #featRaw[i][j]=av_dist_nn
    #j+=1
    
    featRaw[i][j]=sumweights[i]
    j+=1
    
    #featRaw[i][j]=np.dot(weights[i],align_nn)/sumweights[i]
    #j+=1
    
    #featRaw[i][j]=np.dot(weights[i],np.power(dist_nn-av_dist_nn,2))/sumweights[i]
    #j+=1
    
    #featRaw[i][j]=np.dot(weights[i],np.power(angle_nn-featRaw[i][1],2))/sumweights[i]
    #j+=1
    
    #featRaw[i][j]=np.dot(weights[i],np.power(align_nn-featRaw[i][2],2))/sumweights[i]
    #j+=1
    
    sinav = np.dot(weightsKD[i],s2angle)/sumweightsKD[i]
    cosav = np.dot(weightsKD[i],c2angle)/sumweightsKD[i]
    featRaw[i][j]=(sinav**2+cosav**2)**0.5


from sklearn.cluster import KMeans



model = KMeans(n_clusters=2)
model.fit(featRaw)
labels = model.labels_

plt.scatter(featRaw[:, 0], featRaw[:, 1], c=labels.astype(np.float),s=0.1)
