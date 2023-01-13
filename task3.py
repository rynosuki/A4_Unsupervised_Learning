import numpy as np
import matplotlib.pyplot as plt
import sammon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import warnings

###
# Datasets used:
# Balance Scale
# https://www.openml.org/search?type=data&status=any&id=41077
#
# Monks problem
# https://www.openml.org/search?type=data&sort=runs&status=active&id=334
#
# Tic Tac Toe
# https://www.openml.org/search?type=data&sort=runs&status=active&id=50
#
#
# First picture legend:
# Sammon | PCA | t-SNE
#
# Second picture legend:
# BKmeans | KMeans | Agglomerative
#
###


def bkmeans(X, k, iter):
  currentCircles = np.zeros([len(X[:,0])], dtype="int64")
  count = 1
  
  for _ in range(k):
    arrayIndices = np.flatnonzero(currentCircles == np.bincount(currentCircles).argmax())
    curX = X[arrayIndices]
    
    kmeans = KMeans(n_clusters=2, max_iter=iter).fit(curX)
    labels = kmeans.labels_
    
    for i in range(len(arrayIndices)):
      if labels[i] == 1:
        currentCircles[arrayIndices[i]] = count
    count += 1
  return currentCircles

def drtech(X1, X2, X3):
  fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3)
  data = sammon.sammon(X1[:,:4], 100, 3e-2, 0.7)
  ax1.scatter(data[:,0], data[:,1], c = X1[:,4], s = 1)
  
  data = sammon.sammon(X2[:,:20], 100, 3e-2, 0.7)
  ax4.scatter(data[:,0], data[:,1], c = X2[:,20], s = 1)
  
  data = sammon.sammon(X3[:,:9], 100, 3e-2, 0.7)
  ax7.scatter(data[:,0], data[:,1], c = X3[:,9], s = 1)
  
  data = PCA(n_components=2).fit_transform(X1[:,:4])
  ax2.scatter(data[:,0], data[:,1], c = X1[:,4], s = 1)
   
  data = PCA(n_components=2).fit_transform(X2[:,:20])
  ax5.scatter(data[:,0], data[:,1], c = X2[:,20], s = 1)
  
  data = PCA(n_components=2).fit_transform(X3[:,:9])
  ax8.scatter(data[:,0], data[:,1], c = X3[:,9], s = 1)
  
  data1 = TSNE(learning_rate="auto").fit_transform(X1[:,:4])
  ax3.scatter(data1[:,0], data1[:,1], c = X1[:,4], s = 1)  
  
  data2 = TSNE(learning_rate="auto").fit_transform(X2[:,:20])
  ax6.scatter(data2[:,0], data2[:,1], c = X2[:,20], s = 1)
  
  data3 = TSNE(learning_rate="auto").fit_transform(X3[:,:9])
  ax9.scatter(data3[:,0], data3[:,1], c = X3[:,9], s = 1) 
  
  plt.show()
  return [data1, data2, data3] 

def clusteringtech(data):
  fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3)
  
  DATA = bkmeans(data[0], 5, 100)
  ax1.scatter(data[0][:,0], data[0][:,1], c = DATA, cmap = "Accent", s = 1)
  DATA = bkmeans(data[1], 5, 100)
  ax4.scatter(data[1][:,0], data[1][:,1], c = DATA, cmap = "Accent", s = 1)
  DATA = bkmeans(data[2], 5, 100)
  ax7.scatter(data[2][:,0], data[2][:,1], c = DATA, cmap = "Accent", s = 1)
    
  DATA = KMeans(n_clusters=5).fit(data[0])
  ax2.scatter(data[0][:,0], data[0][:,1], c = DATA.labels_, cmap = "Accent", s = 1)
  DATA = KMeans(n_clusters=5).fit(data[1])
  ax5.scatter(data[1][:,0], data[1][:,1], c = DATA.labels_, cmap = "Accent", s = 1)
  DATA = KMeans(n_clusters=5).fit(data[2])
  ax8.scatter(data[2][:,0], data[2][:,1], c = DATA.labels_, cmap = "Accent", s = 1)

  DATA = AgglomerativeClustering(n_clusters=5).fit(data[0])
  ax3.scatter(data[0][:,0], data[0][:,1], c = DATA.labels_, cmap = "Accent", s = 1)
  DATA = AgglomerativeClustering(n_clusters=5).fit(data[1])
  ax6.scatter(data[1][:,0], data[1][:,1], c = DATA.labels_, cmap = "Accent", s = 1)
  DATA = AgglomerativeClustering(n_clusters=5).fit(data[2])
  ax9.scatter(data[2][:,0], data[2][:,1], c = DATA.labels_, cmap = "Accent", s = 1)
  plt.show()

def main():
  warnings.filterwarnings("ignore")
  balance = np.genfromtxt("balance_scale.csv", delimiter=",")[1:,:]
  climate = np.genfromtxt("climate.csv", delimiter=",")[:,:]
  tic = np.genfromtxt("tictactoe.csv", delimiter=",")[1:,:]
  
  data = drtech(balance, climate, tic)
  clusteringtech(data)
main()