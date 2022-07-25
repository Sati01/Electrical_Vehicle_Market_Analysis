# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Step 3: Collecting data
data = pd.read_csv('/content/Mcdonalds.csv')

data.head(3)

# Step 4: Exploring data
data_x=data[1:11]
data_x=(data_x=="Yes")+0
print(data_x.mean(), '\n')

pca=PCA().fit(data_x)
df_pca=pca.transform(data_x)

print('Variance:', list((pca.explained_variance_ratio_*100).round(4)))

plt.figure(figsize = (10,10))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(df_pca[:, 0], df_pca[:, 1])
plt.show()

# Step 5: Extracting Segments
# 5.1
df_K_means=[]
for i in range(1, 9):
    km=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = km.fit_predict(data_x)
    df_K_means.append(km.inertia_)

plt.bar(range(1, 9), df_K_means)
plt.xlabel('Number of segments')
plt.ylabel('WCSS')
plt.show()

MDk4=df_K_means[:4]

import plotly.figure_factory as ff

names = data_x.columns
fig = ff.create_dendrogram(data_x, orientation='left', labels=names)
fig.update_layout(width=800, height=800)

fig.show()

def biplot(score,coeff,pcax,pcay,labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5)
        if labels is None:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()

biplot(df_pca[:,0:8],pca.components_,1,2,labels=data_x.columns)

