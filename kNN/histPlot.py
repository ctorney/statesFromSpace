

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

#valsDF=valsDF[0:200]
N = len(valsDF)

NFeats = 2 # number of features of the herd (mean nn, variance in nn, mean of 3 neighbors, circular variance)
featRaw = np.empty((N,NFeats)) # 250 images x 4096 pixel values - raw features


dist_knn = valsDF['dist_knn'].values
angle_knn = valsDF['angle_knn'].values
align_knn = valsDF['align_knn'].values

plt.figure()
plt.hist(angle_knn,bins=100)

plt.figure()
plt.hist(align_knn,bins=200)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dist_knn,angle_knn,align_knn,s=0.1)



