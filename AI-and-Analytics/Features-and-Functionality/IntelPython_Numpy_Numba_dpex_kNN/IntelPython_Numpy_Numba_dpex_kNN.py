#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================
# Copyright © 2022 Intel Corporation
# 
# SPDX-License-Identifier: MIT
# =============================================================


# # Simple k-NN classification with numba_dpex IDP optimization
# 
# This sample shows how to recieve the same accuracy of the k-NN model classification by using numpy, numba and numba_dpex. The computetaion are performed using wine dataset.
# 
# Let's start with general inports used in the whole sample.

# In[ ]:


import math
from collections import Counter
import numpy as np
import pandas as pd


# ## Data preparation
# 
# Then, let's download the dataset and prepare it for future computations.
# 
# We are using the wine dataset available in the sci-kit learn library. For our purposes, we will be using only 2 features: alcohol and malic_acid.
# 
# So first we need to load the dataset and create DataFrame from it. Later we will limit the DataFrame to just target and 2 classes we choose for this problem.

# In[ ]:


from sklearn.datasets import load_wine

data = load_wine()
# Convert loaded dataset to DataFrame 
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)

# Limit features to 2 selected for this problem 
df = df[['target', 'alcohol', 'malic_acid']]

# Show top 5 values from the limited dataset
df.head()


# We are planning to compare the results of the numpy, namba and IDP numba_dpex so we need to make sure that the results are reproducible. We can do this through the use of a random seed function that initializes a random number generator.

# In[ ]:


np.random.seed(42)


# The next step is to prepare the dataset for training and testing. To do this, we randomly divided the downloaded wine dataset into a training set (containing 90% of the data) and a test set (containing 10% of the data). 
# 
# In addition, we take from both sets (training and test) data *X* (features) and label *y* (target).

# In[ ]:


# we are using 10% of the data for the testing purpose
train_sample_idx = np.random.choice(df.index, size=int(df.shape[0]*0.9), replace=False)
train_data, test_data = df.iloc[train_sample_idx], df.drop(train_sample_idx)

# get features and label from train/test data
X_train, y_train = train_data.drop('target', axis=1), train_data['target']
X_test, y_test = test_data.drop('target', axis=1), test_data['target']


# ## NumPy k-NN
# 
# Now, it's time to implenet the first version of k-NN function using NumPy.
# 
# First, let's create simple euqlidesian distance function. We are taking positions form the provided vectors, counting the squares of the individual differences between the positions, and then drawing the root of their sum for the whole vectors (remember that the vectors must be of equal length).

# In[ ]:


def distance(vector1, vector2):
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist


# Then, the k-nearest neighbour algorithm itself.
# 
# 1. We are starting by defining a container for predictions the same size as a test set.
# 2. Then, for each row in the test set, we calculate distances between then and every training record.
# 3. We are sorting training datasets based on calculated distances
# 4. Choose k of the first elements in the sorted training list.
# 5. We are counting labels appearances
# 6. The most common label is set as a prediction. 

# In[ ]:


def knn(X_train, y_train, X_test, k):
    # 1. Prepare container for predictions
    predictions = np.zeros(X_test.shape[0])
    X_test.reset_index(drop=True, inplace=True)
    
    for index, row in X_test.iterrows():
        # 2. Calculate distances
        inputs = X_train.copy()
        inputs['distance'] = inputs.apply(distance, vector2=row, axis=1)
        inputs = pd.concat([inputs, y_train], axis=1)
        
        # 3. Sort based on distance
        inputs = inputs.sort_values('distance', ascending=True)
        
        # 4. Choose k if the first elements
        neighbors = inputs.head(k)
        classes = neighbors['target'].tolist()
        
        # 5. Count labels appearances
        majority_count = Counter(classes)
        
        # 6. Choose most common label
        predictions[index] = majority_count.most_common(1).pop()[0]
        
    return predictions


# Let's use our prepared knn function on the wine dataset. Let's assume `k = 3`.
# The accuracy of the predicted labels is measured as the mean of the truly predicted values.

# In[ ]:


# knn = KNN(3)
predictions = knn(X_train, y_train, X_test, 3)
true_values = y_test.to_numpy()
accuracy = np.mean(predictions == true_values)
print('Numpy accuracy:', accuracy)


# ## Numba k-NN
# 
# Now, let's move to the numba implementation of the k-NN algorithm. We will start the same, by defining the distance function and importing the necessary packages.
# 
# For numba implementation, we are using the core functionality which is `numba.jit()` decorator.
# 
# We are starting with defining the distance function. Like before it is a euclidean distance. For additional optimization we are using `np.linalg.norm`.

# In[ ]:


import numba

@numba.jit(nopython=True)
def euclidean_distance_numba(vector1, vector2):
    dist = np.linalg.norm(vector1-vector2)
    return dist


# The next step is to implement the k-NN algorithm. Like before, there is `numba.jit()` decorator used. Other steps for the algorithm are the same as for the NumPy example.

# In[ ]:


@numba.jit(nopython=True)
def knn_numba(X_train, y_train, X_test, k):
    # 1. Prepare container for predictions
    predictions = np.zeros(X_test.shape[0])
    for x in np.arange(X_test.shape[0]):
                
        # 2. Calculate distances
        inputs = X_train.copy()
        distances = np.zeros((inputs.shape[0], 1))
        for i in np.arange(inputs.shape[0]):
            distances[i] = euclidean_distance_numba(inputs[i], X_test[x])
        
        labels = y_train.copy()
        labels = labels.reshape((labels.shape[0],1))

        # add labels column
        inputs = np.hstack((inputs, labels))
        # add distance column
        inputs = np.hstack((inputs, distances))

        # 3. Sort based on distance
        inputs = inputs[inputs[:,3].argsort()]
        # 4. Choose k if the first elements
        # 2nd columns contains classes, select first k values
        neighbor_classes = inputs[:, 2][:k]

        # 5. Count labels appearances
        counter = {}
        for item in neighbor_classes:
            if item in counter:
                counter[item] = counter.get(item) + 1
            else:
                counter[item] = 1
        counter_sorted = sorted(counter)
        
        # 6. Choose most common label
        predictions[x] = counter_sorted[0]
    return predictions


# Similarly, as in the NumPy example, we are testing implemented method for the `k = 3`. 
# 
# The accuracy of the method is the same as in the NumPy implementation.

# In[ ]:


# knn(3) using numba.jit() decorator
predictions = knn_numba(X_train.values, y_train.values, X_test.values, 3)
true_values = y_test.to_numpy()
accuracy = np.mean(predictions == true_values)
print('Numba accuracy:', accuracy)


# ## Numba_dpex k-NN
# 
# Numba_dpex implementation use `numba_dpex.kernel()` decorator. For more information about programming, SYCL kernels go to: https://intelpython.github.io/numba-dpex/latest/user_guides/kernel_programming_guide/index.html.
# 
# Calculating distance is like in the NumPy example. We are using Euclidean distance. Later, we create the queue of the neighbours by the calculated distance and count in provided *k* votes for dedicated classes of neighbours.
# 
# In the end, we are taking a class that achieves the maximum value of votes and setting it for the current global iteration.

# In[ ]:


import   numba_dpex

@numba_dpex.kernel
def knn_numba_dpex(train, train_labels, test, k, predictions, votes_to_classes_lst):
    i = numba_dpex.get_global_id(0)
    queue_neighbors = numba_dpex.private.array(shape=(3, 2), dtype=np.float64)
    
    for j in range(k):
        x1 = train[j][0]
        x2 = test[i][0]

        distance = 0.0
        diff = x1 - x2
        distance += diff * diff
        dist = math.sqrt(distance)

        queue_neighbors[j, 0] = dist
        queue_neighbors[j, 1] = train_labels[j]
        
    for j in range(k):
        new_distance = queue_neighbors[j, 0]
        new_neighbor_label = queue_neighbors[j, 1]
        index = j

        while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
            queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
            queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

            index = index - 1

            queue_neighbors[index, 0] = new_distance
            queue_neighbors[index, 1] = new_neighbor_label

    for j in range(k, len(train)):
        x1 = train[j][0]
        x2 = test[i][0]

        distance = 0.0
        diff = x1 - x2
        distance += diff * diff
        dist = math.sqrt(distance)
        
        if dist < queue_neighbors[k - 1][0]:
            queue_neighbors[k - 1][0] = dist
            queue_neighbors[k - 1][1] = train_labels[j]
            new_distance = queue_neighbors[k - 1, 0]
            new_neighbor_label = queue_neighbors[k - 1, 1]
            index = k - 1

            while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
                queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
                queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

                index = index - 1

                queue_neighbors[index, 0] = new_distance
                queue_neighbors[index, 1] = new_neighbor_label

    votes_to_classes = votes_to_classes_lst[i]

    for j in range(len(queue_neighbors)):
        votes_to_classes[int(queue_neighbors[j, 1])] += 1

    max_ind = 0
    max_value = 0

    for j in range(3):
        if votes_to_classes[j] > max_value:
            max_value = votes_to_classes[j]
            max_ind = j

    predictions[i] = max_ind


# Next, like before, let's test the prepared k-NN function.
# 
# In this case, we will need to provide the container for predictions: `pedictions_numba` and the container for votes per class: `votes_to_classes_lst` (the container size is 3, as we have 3 classes in our dataset).
# 
# We are running a prepared k-NN function using `dctl.device_context()`, which allows us to select a divice. For more information, go to: https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/device_selection.html.

# In[ ]:


import dpctl

predictions_numba = np.empty(len(X_test.values))
# we have 3 classes
votes_to_classes_lst = np.zeros((len(X_test.values), 3))
with dpctl.device_context("opencl:cpu:0"):
    knn_numba_dpex[len(X_test.values), numba_dpex.DEFAULT_LOCAL_SIZE](X_train.values, y_train.values, X_test.values, 3, predictions_numba, votes_to_classes_lst)


# Like before, let's measure the accuracy of the prepared implementation. It is measured as the number of well-assigned classes for the test set. The final result is the same for all: NumPy, numba and numba-dpex implementations.

# In[ ]:


true_values = y_test.to_numpy()
accuracy = np.mean(predictions_numba == true_values)
print('Numba_dpex accuracy:', accuracy)


# In[ ]:


print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")

