from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import time


class Clustering:

    # Determine an approximate n_cluster for the clustering algorithm
    def elbow_chart(self, df, model_type):
        model = KMeans()  # create a model for elbow method
        if (model_type == 'AGNES'):
            model = AgglomerativeClustering()

        visualizer = KElbowVisualizer(model, k=(1, 12))

        visualizer.fit(df)
        visualizer.show()
        return visualizer.elbow_value_

    def kmeans_clustering(self, df, pca, colors, n_clusters):
        model = KMeans(n_clusters=n_clusters, random_state=0)

        model = model.fit(pca)
        y = model.predict(pca)

        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(pca[y == i, 0], pca[y == i, 1], s=50,
                        c=colors[i], label='Cluster #' + str(i))
        plt.scatter(model.cluster_centers_[:, 0],
                    model.cluster_centers_[:, 1], s=200, c='black', label='Centroids')
        plt.title('KMeans Cluster')
        plt.grid()
        plt.legend()
        plt.show()

    def agnes_clustering(self, df, pca, colors, n_clusters):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        model.fit_predict(pca)

        plt.scatter(pca[:, 0], pca[:, 1], c=model.labels_, cmap='rainbow')
        plt.title('AGNES Clustering')
        plt.grid()
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.title('Dendogram, AGNES Clustering')
        shc.dendrogram((shc.linkage(pca, method='ward')))
        plt.grid()
        plt.show()

    def dbscan_clustering(self, df, pca, eps, min_samples):

        # first scale the pca results
        scaler = StandardScaler()
        scaler = scaler.fit(pca)
        scaled_p = scaler.transform(pca)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_p)
        labels = db.labels_
        real_n_clusters_ = len(set(labels)) - (1 if - 1 in labels else 0)
        n_clusters_ = len(set(labels))
        print('number of clusters in pca-DBSCAN: ' + str(real_n_clusters_))

        plt.scatter(scaled_p[:,0], scaled_p[:,1], c=labels, s=60, edgecolors='black', cmap='rainbow')
        plt.title('DBSCAN Clustering, estimated number of clusters = 2')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def eps_selection(self, pca):
        # first scale the pca results
        scaler = StandardScaler()
        scaler = scaler.fit(pca)
        scaled_p = scaler.transform(pca)

        # apply 2-NN algorithm to find distances
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(scaled_p)
        distances, indices = nbrs.kneighbors(scaled_p)

        # plot the distances and observe an approximate epsilon value
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.plot(distances)
        plt.show()
