import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics

def sammon(X, iter, err, alpha):
  nPoints = len(X)
  Y = np.random.normal(0.0, 1.0, [nPoints, 2])

  # Calculate constant (used in 2 and 4)
  constant = 0
  inputDistances = metrics.pairwise_distances(X)
  for i in range(nPoints):
    for j in range(nPoints):
      if i < j:
        constant += inputDistances[i,j]

  for _ in range(iter):
    outputDistances = metrics.pairwise_distances(Y)
    E = 0
    for i in range(nPoints):
      for j in range(nPoints):
        if i < j:
          if inputDistances[i,j] < 0.000001:
            E += ((outputDistances[i,j] - inputDistances[i,j])**2) / 0.00001
          else:
            E += ((outputDistances[i,j] - inputDistances[i,j])**2) / inputDistances[i,j]
    E = E / constant
    if E < err:
      return Y

    for i in range(nPoints):
      pd1, pd2 = 0, 0
      for j in range(nPoints):
        if j != i:
          denominator = outputDistances[i,j] * inputDistances[i,j]
          if denominator < 0.0001:
            denominator = 0.001
          pd1 += ((inputDistances[i,j] - outputDistances[i,j]) / denominator) * (Y[i] - Y[j])
          pd2 += ((inputDistances[i,j] - outputDistances[i,j]) - ((Y[i] - Y[j])**2 / outputDistances[i,j]) * (1 + ((inputDistances[i,j] - outputDistances[i,j]) / outputDistances[i,j]))) / denominator
      pd = np.divide((-2/constant) * pd1, np.abs((-2/constant) * pd2))
      Y[i] = Y[i] - alpha * pd
      curE = E
  return Y

def plot_3d(points, points_color, title):
  x, y, z = points.T
  fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
  )
  fig.suptitle(title)
  ax.scatter(x, y, z, c=points_color, s=50)
  ax.view_init(azim=-60, elev=9)
  plt.show()

def main():
  X, target = datasets.make_s_curve(n_samples=200)
  reduced = sammon(X, 100, 3e-2, 0.7)
  plt.show()
  plot_3d(X, target, "Original S-curve samples")
  plt.scatter(reduced[:, 0], reduced[:, 1], c=target, cmap='rainbow', s = 1)
  plt.show()

if __name__ == "__main__":
  main()