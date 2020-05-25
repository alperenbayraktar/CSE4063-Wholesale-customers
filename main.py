import pandas as pd
import seaborn as sns
import collections
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import PreProcessing
from graphs import Graphs
import time
from clustering import Clustering
from frequent_pattern import FrequentPattern
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from mlxtend.preprocessing import TransactionEncoder


#cols = ["Channel","Region","Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
df = pd.read_csv("test.csv",header=None,names=range(11))
all_labels = pd.unique(df.values.ravel('K'))
all_labels = all_labels[:-1]
#df = df.replace(np.nan, '', regex=True)
print(all_labels)
pp = PreProcessing()
gr = Graphs()
fp = FrequentPattern()
clustering  = Clustering()
#pp.categorize(df)
#gr.get_visual_data(df)

#df = df.drop(['Channel'], axis=1)
#df = df.drop(['Region'], axis=1)
start_time = time.time()
data_list = df.to_numpy()
new_data_list = []
for entry in data_list:
    new_entry = []
    for element in entry:
        if((element in all_labels) == True):
            new_entry.append(element)
    new_data_list.append(new_entry)
end_time = time.time()
print(end_time-start_time)
#print(new_data_list) 

te = TransactionEncoder()
te_ary = te.fit(new_data_list).transform(new_data_list)
new_df = pd.DataFrame(te_ary, columns=te.columns_)
print(new_df)

min = 0.001


frequent_itemsets_apriori = fp.apriori(new_df,min,1)
print(frequent_itemsets_apriori.sort_values(by=['support'],ascending=False))

frequent_itemsets_fpgrowth = fp.fpgrowth(new_df,min,1)
print(frequent_itemsets_fpgrowth.sort_values(by=['support'],ascending=False))

frequent_itemsets_eclat =  fp.eclat(frequent_itemsets_apriori,min)
print(frequent_itemsets_eclat.sort_values(by=['support'],ascending=False))

list_of_datasets = [frequent_itemsets_apriori, frequent_itemsets_fpgrowth, frequent_itemsets_eclat]
min_supports = [0.100, 0.150, 0.200]
def get_all_graphs(list_of_datasets,min_supports):
    for dataset in list_of_datasets:
        gr.get_single_graph(dataset)
        for support in min_supports:
            gr.get_single_graph(dataset,support)
    gr.get_multiple_graph(list_of_datasets[0], list_of_datasets[1], list_of_datasets[2])
    for support in min_supports:
        gr.get_multiple_graph(list_of_datasets[0], list_of_datasets[1], list_of_datasets[2],filter_support=support)

get_all_graphs(list_of_datasets,min_supports)

df = pd.read_csv("data.csv")
df.drop(['Channel', 'Region'], axis=1)
colors = ['red', 'green', 'blue', 'orange']
pca = PCA(n_components=2, random_state=1)
pca_result = pca.fit_transform(df)
# Determine the number of clusters for KMeans and AGNES algorithms
n_clusters_kmeans = clustering.elbow_chart(df, model_type='Kmeans')
n_clusters_agnes = clustering.elbow_chart(df, model_type='AGNES')
clustering.kmeans_clustering(df, pca_result, colors, n_clusters_kmeans)
clustering.agnes_clustering(df, pca_result, colors, n_clusters_agnes)
# Selecting epsilon for DBSCAN by running an NearestNeighbours algorihm
# The optimal eps value is 0.3 as can be shown from the plot
clustering.eps_selection(pca_result)
clustering.dbscan_clustering(df, pca_result, eps=0.3, min_samples=15)
