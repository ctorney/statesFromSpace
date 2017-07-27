
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
import gap


posfilename = '../data/node_values.csv'


valsDF = pd.read_csv(posfilename) 
N = len(valsDF)

NFeats = 2 # number of features of the herd (mean nn, variance in nn, mean of 3 neighbors, circular variance)
featRaw = np.empty((N,NFeats)) # 250 images x 4096 pixel values - raw features


distances = np.zeros((N,N))
xvals = valsDF['x'].values
yvals = valsDF['y'].values
dist_nn = valsDF['dist_nn'].values
angle_nn = valsDF['angle_knn'].values
align_nn = valsDF['align_knn'].values

for i in range(N):
    for j in range(i+1,N):
        distances[i,j]=math.sqrt((xvals[i]-xvals[j])**2+(yvals[i]-yvals[j])**2)
        distances[j,i]=distances[i,j]

length = 0.1

#for length in np.arange(1,101,1):
#    
#    weights = np.zeros_like(distances)
#    #weights[distances<length]=1.0
#    weights = np.exp(-np.power(distances/length,2))
#    sumweights=np.sum(weights,axis=1)
#    featRaw[:,0]=angle_nn
#    featRaw[:,1]=align_nn
#    for i in range(N):
#        j=0
#        
#        featRaw[i][j]=np.dot(weights[i],angle_nn)/sumweights[i]
#        j+=1
#        
#        featRaw[i][j]=np.dot(weights[i],align_nn)/sumweights[i]
#        j+=1
#    
#    gp=gap.gap(featRaw,ks=np.array([3]))
#    print(length,gp[0])
# # Visualisation
# 
# Now that we have a set of features (Naive and HOGs) we will use linear (PCA) and non-linear (t-SNE) dimensionality reduction techniques to visualise the data and see whether we can detect some structure in the data.
length=50
weights = np.zeros_like(distances)
#weights[distances<length]=1.0
weights = np.exp(-np.power(distances/length,2))
sumweights=np.sum(weights,axis=1)
featRaw[:,0]=angle_nn
featRaw[:,1]=align_nn
for i in range(N):
    j=0
    
    featRaw[i][j]=angle_nn[i]#np.dot(weights[i],angle_nn)/sumweights[i]
    j+=1
    
    featRaw[i][j]=align_nn[i]#np.dot(weights[i],align_nn)/sumweights[i]
    j+=1


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from sklearn.metrics import calinski_harabaz_score

nk=3
kmm = KMeans(n_clusters=nk)

kmm.fit(featRaw)

labels = kmm.labels_
valsDF['class']=labels
valsDF.to_csv(posfilename,index=False)
#silhouette_score(featRaw,labels)
#labels[ps[:,0]<sw]=0
#labels[ps[:,0]>=sw]=1

plt.figure()
plt.scatter(featRaw[labels==0,0],featRaw[labels==0,1], c=labels.astype(np.float))

plt.figure()
plt.scatter(featRaw[labels==2,0],featRaw[labels==2,1])


plt.hist(featRaw[labels==0,0],range=[0,1],bins=100)
plt.hist(featRaw[labels==1,0],range=[0,1],bins=100)
plt.hist(featRaw[labels==2,0],range=[0,1],bins=100)

plt.hist(featRaw[labels==0,1],range=[0,1],bins=100)
plt.hist(featRaw[labels==1,1],range=[0,1],bins=100)
plt.hist(featRaw[labels==2,1],range=[0,1],bins=100)


#gapstats = gap.gap(featRaw)

#featRaw2=featRaw[::2,:]