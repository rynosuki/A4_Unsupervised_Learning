from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

def bkmeans(X, k, iter):
  currentCircles = np.zeros([len(X[:,0])], dtype="int64")
  count = 1
  
  for _ in range(k):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=currentCircles)
    
    arrayIndices = np.flatnonzero(currentCircles == np.bincount(currentCircles).argmax())
    curX = X[arrayIndices]
    
    kmeans = KMeans(n_clusters=2, max_iter=iter).fit(curX)
    labels = kmeans.labels_
    
    for i in range(len(arrayIndices)):
      if labels[i] == 1:
        currentCircles[arrayIndices[i]] = count
    count += 1
  return currentCircles
    
def main():
  X,y = make_blobs(n_samples=1000)
  clusterindices = bkmeans(X, 5, 100)
  print(clusterindices)
  plt.show()

if __name__ == "__main__":
  main()