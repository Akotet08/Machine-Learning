import numpy as np
from sklearn.datasets import make_circles
from sklearn import metrics
import matplotlib.pyplot as plt


def kmeans(data, k):
    m, n = data.shape
    # First, randomly pick k data points for the centers of the initial clusters. Maybe you should use the np.random.choice() function.
    centroids = data[np.random.choice(m, size=k)]
    # Afterwards, initialize the clusters.
    clusters = np.random.choice(k, m)
    
    # Update the clusters until convergence.
    loss = 0
    updated = True
    while updated:
        updated = False
        # Calculate the cluster means.
        cluster_mean = np.zeros((k, n))
        for i in range(k):
            cluster_mean[i] = np.mean(data[np.where(i == clusters)], axis=0)

        # Find out which new cluster each data point belongs to
        new_clusters = np.copy(clusters)
        sum = 0
        for i in range(m):
            min_dis = np.Inf
            for j in range(k):
                distance = np.sum(np.square(data[i] - cluster_mean[j]))
                if(distance < min_dis):
                    min_dis = distance
                    new_clusters[i] = j
            sum += distance
        # Update the cluster allocation. If nothing changes, exit the loop and return the converged result.
        if( not np.array_equal(new_clusters, clusters)):
            updated = True
            clusters = new_clusters
        else:
            loss = sum
    return clusters, loss

if __name__ == '__main__':
    X, y = make_circles(n_samples=1000, noise = 0.1, factor = 0.3, random_state = 10)
    iter = 100 # we run kernel k-means iter times to obtain the best fit
    kms, loss = kmeans(X, k=2)
    best_kms = kms
    min_loss = loss
    for _ in range(iter):
        kms, loss = kmeans(X, k=2)
        if loss < min_loss:
            min_loss = loss
            best_kms = kms
    # Visualize the result of the k-means clustering.
    plt.figure()
    plt.scatter(X[kms == 0, 0], X[kms == 0, 1], color='red') 
    plt.scatter(X[kms == 1, 0], X[kms == 1, 1], color='blue') 
    plt.show()

    print(metrics.normalized_mutual_info_score(y, best_kms))