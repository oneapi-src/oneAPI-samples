#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
=============================================================
Copyright © 2020 Intel Corporation

SPDX-License-Identifier: MIT
=============================================================
'''

# # Daal4py K-Means Clustering Example for Distributed Memory Systems [SPMD mode]

# ## IMPORTANT NOTICE

# When using daal4py for distributed memory systems, the command needed to execute the program should be **executed 
# in a bash shell**. In order to run this example, please download it as a .py file then run the following command (**the number 4 means that it will run on 4 processes**):

# mpirun -n 4 python ./daal4py_Distributed_Kmeans.py

# ## Importing and Organizing Data

# In this example we will be using K-Means clustering to **initialize centroids** and then **use them to cluster the synthetic dataset.**
# 
# Let's start by **importing** all necessary data and packages.

# In[2]:


##### daal4py K-Means Clustering example for Distributed Memory Systems [SPMD Mode] #####
import daal4py as d4p
import pickle
import pandas as pd
import numpy as np


# Now let's **load** in the dataset and **organize** it as necessary to work with our model. For distributed, every file has a unique ID.
# 
# We will also **initialize the distribution engine**.

# In[3]:


d4p.daalinit() #initializes the distribution engine

# organizing variables used in the model for prediction
# each process gets its own data
infile = "./data/distributed_data/daal4py_Distributed_Kmeans_" + str(d4p.my_procid()+1) + ".csv"

# read data
X = pd.read_csv(infile)


# ## Computing and Saving Initial Centroids

# Time to **initialize our centroids!**

# In[4]:


# computing inital centroids
init_result = d4p.kmeans_init(nClusters = 3, method = "plusPlusDense").compute(X)


# To **get initial centroid information and save it** to a file:

# In[5]:


# retrieving and printing inital centroids
centroids = init_result.centroids
print("Here our centroids:\n\n\n", centroids, "\n")

centroids_filename = './models/kmeans_clustering_initcentroids_'+  str(d4p.my_procid()+1) + '.csv'

# saving centroids to a file
pickle.dump(centroids, open(centroids_filename, "wb"))


# Now let's **load up the centroids** and look at them.

# In[6]:


# loading the initial centroids from a file
loaded_centroids = pickle.load(open(centroids_filename, "rb"))
print("Here is our centroids loaded from file:\n\n",loaded_centroids)


# # Assign The Data to Clusters and Save The Results

# Let's **assign the data** to clusters.

# In[7]:


# compute the clusters/centroids
kmeans_result = d4p.kmeans(nClusters = 3, maxIterations = 5, assignFlag = True).compute(X, init_result.centroids)


# To **get Kmeans result objects** (assignments, centroids, goalFunction [deprecated], nIterations, and objectiveFunction):

# In[8]:


# retrieving and printing cluster assignments
assignments = kmeans_result.assignments
print("Here is our cluster assignments for first 5 datapoints: \n\n", assignments[:5])


# Now let's **export the cluster assignments** to a **CSV file**. We will also **stop the distribution engine.**

# In[9]:


# now export the results to a CSV file
results_filename = "./results/daal4py_Distributed_Kmeans_results_" + str(d4p.my_procid()+1) + ".csv"
np.savetxt(results_filename, assignments, delimiter=",")

d4p.daalfini() # stops the distribution engine
print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')

