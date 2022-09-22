import numpy as np
from sklearn.datasets import make_circles
from sklearn import metrics
import matplotlib.pyplot as plt

def kernel_kmeans(data, k, s, kernel = 'gaussian'):
    # The variable s refers to the value of sigma in the Gaussian kernel.
    m, n = data.shape
    # First, randomly pick k data points for the centers of the initial clusters. Maybe you should use the np.random.choice() function.
    centroids = data[np.random.choice(range(m),k)]
    # Afterwards, initialize the clusters.
    mat = np.zeros((m, k))
    for j in range(k):
        mat[:, j] = np.linalg.norm(data - centroids[j], axis=1)

    clusters = np.argmin(mat, axis=1)
    
    # Pre-calculate kernel values and save it in a matrix.
    pariwise_dis = metrics.pairwise_distances(data)
    # Gaussian kernel
    gaussian = np.exp((-np.square(pariwise_dis)) / (2*(s**2)))

    # Update the clusters until convergence.
    loss = 0
    updated = True
    while updated:
        updated = False
        # Find out which new cluster each data point belongs to
        for i in range(k):
            second_term = np.sum(gaussian[:, clusters== i], axis=1) / np.sum(clusters == i)
            third_term = np.sum(gaussian[clusters == i][:, clusters == i]) / (np.sum(clusters == i)**2)
            mat[:, i] = 1 - 2*second_term  + third_term
        
        new_clusters = np.argmin(mat, axis=1)
        # Update cluster allocation. If nothing changes, exit the loop and return the converged result.
        if( not np.array_equal(new_clusters,clusters)):
            updated = True
            clusters = new_clusters
        else:
            loss = np.sum(np.min(mat, axis=1))
    return clusters, loss


if __name__=='__main__':
    X, y = make_circles(n_samples=1000, noise = 0.1, factor = 0.3, random_state = 10) 
    iter = 10 # we run kernel k-means iter times to obtain the best fit
    kms, loss = kernel_kmeans(X, k=2, s=0.5)
    best_kms = kms
    min_loss = loss
    for _ in range(iter):
        kms, loss = kernel_kmeans(X, k=2, s=0.5)
        if loss < min_loss:
            min_loss = loss
            best_kms = kms

    # Visualize the result of the kernel k-means clustering.
    plt.figure()
    plt.scatter(X[best_kms == 0, 0], X[best_kms == 0, 1], color='red') 
    plt.scatter(X[best_kms == 1, 0], X[best_kms == 1, 1], color='blue') 
    plt.show()

    print(metrics.normalized_mutual_info_score(y, best_kms))

