#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation Using Python

# ### Importing Modules

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

import warnings
warnings.filterwarnings('ignore')


# In[12]:


data = pd.read_csv('Mall_Customers.csv')
data.head()


# In[13]:


X = data.iloc[:, [3, 4]].values


# ### Use Dendrogram to find optimal number of clusters

# In[15]:


dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()


# ### Perform Hierarchical Clustering

# In[19]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)


# In[20]:


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# You can Find Project on <a href="https://github.com/Vyas-Rishabh/Customer_Segmentation_Using_Python"><b>GitHub.</b></a>
