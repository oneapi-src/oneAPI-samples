#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
=============================================================
Copyright © 2020 Intel Corporation
 
SPDX-License-Identifier: MIT
=============================================================
'''

# # IntelPython Getting Started Example for Shared Memory Systems

# ## Importing and Organizing Data

# In this example we will be predicting **prices of houses in Boston** based on the features of each house.
# 
# Let's start by **importing** all necessary data and packages.

# In[2]:


##### Linear regression example for shared memory systems #####
import daal4py as d4p
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle


# Now let's **load** in the dataset and **organize** it as necessary to work with our model.

# In[3]:


# loading in the data
data = load_boston()

# organizing variables used in the model for prediction
X = data.data # house characteristics
y = data.target[np.newaxis].T # house price

# splitting the data for training and testing, with a 25% test dataset size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1693)


# ## Training and Saving the Model

# Let's **train our model** and look at the model's features!

# In[4]:


# training the model for prediction
train_result = d4p.linear_regression_training().compute(X_train, y_train)


# To **get training model information** and **save it to a file**:

# In[5]:


# retrieving and printing training model
model = train_result.model
print("Here's our model:\n\n\n", model , "\n")

model_filename = './models/linear_regression_batch.sav'

# saving model to a file
pickle.dump(model, open(model_filename, "wb"))


# Now let's **load up the model** and look at one of the model's features.

# In[6]:


# loading the training model from a file
loaded_model = pickle.load(open(model_filename, "rb"))
print("Here is one of our loaded model's features: \n\n", loaded_model.Beta)


# ## Making a Prediction and Saving the Results

# Time to **make a prediction!**

# In[7]:


# now predicting the target feature(s) using the trained model
y_pred = d4p.linear_regression_prediction().compute(X_test, loaded_model).prediction 


# Now let's **export the results to a CSV file**.

# In[8]:


np.savetxt("./results/linear_regression_batch_results.csv", y_pred, delimiter =  ",")
print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")

