import numpy as np
from numpy.random import uniform

# Euclidean Distance
def eucl_dis(point, data):
  # axis 1 because data is an m by n 
  # point - data[0] = result_1
  # point - data[1] = result_2
  # point - data[n] = result_n
  # is [result_1, result_2, result_n]
  return np.sqrt(np.sum((point-data)**2, axis=1))

class KMEANS:
  def __init__(self, n_clusters=4, max_iter=200):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.cluster_centers_ = None
    self.n_iter_ = None

  def fit(self, data):
    # Randomly select centroid start points, uniformly distributed across the domain of the dataset
    min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
    self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

    prev_centroids = None
    curr_iter = 0

    # .any() returns True if an element in an iterable is True
    while (curr_iter < self.max_iter) and np.not_equal(self.centroids, prev_centroids).any():
      sorted_points = [[] for _ in range(self.n_clusters)]
      for x in X_train:
        dists = eucl_dis(x, self.centroids)
        centroid_idx = np.argmin(dists)
        sorted_points[centroid_idx].append(x)

      # update new centroids (mean) of each cluster
      prev_centroids = self.centroids
      self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
      for i, centroid in enumerate(self.centroids):
        if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
            self.centroids[i] = prev_centroids[i]
      curr_iter += 1

      self.cluster_centers_ = self.centroids
      self.n_iter_ = curr_iter
      
      
if __name__ == "__main__":
  import seaborn as sns
  from sklearn.datasets import make_blobs
  import matplotlib.pyplot as plt
  from sklearn.preprocessing import StandardScaler
  centers = 5
  X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
  X_train = StandardScaler().fit_transform(X_train)
  sns.scatterplot(x=[X[0] for X in X_train],
                  y=[X[1] for X in X_train],
                  hue=true_labels,
                  palette="deep",
                  legend=None
                  )
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()
  
  kmeans = KMEANS(n_clusters=5)
  kmeans.fit(X_train)
  print(kmeans.cluster_centers_)
  print(kmeans.n_iter_)
  
  # View results
  sns.scatterplot(x=[X[0] for X in X_train],
                  y=[X[1] for X in X_train],
                  hue=true_labels,
                  palette="deep",
                  legend=None
                  )
  plt.plot([x for x, _ in kmeans.cluster_centers_],
           [y for _, y in kmeans.cluster_centers_],
           '+',
           markersize=10,
           )
  plt.show()
