#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================
# Copyright © 2020 Intel Corporation
# 
# SPDX-License-Identifier: MIT
# =============================================================


# # IntelPython Getting Started Example for daal4py

# ## Importing and Organizing Data

# In this example we will be predicting **prices of houses in California** based on the features of each house.
# 
# Let's start by **importing** all necessary data and packages.

# In[12]:


##### Linear regression example for shared memory systems #####
import daal4py as d4p
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib


# Now let's **load** in the dataset and **organize** it as necessary to work with our model.

# In[13]:


# loading in the data
data = fetch_california_housing()

# organizing variables used in the model for prediction
X = data.data # house characteristics
y = data.target[np.newaxis].T # house price

# splitting the data for training and testing, with a 25% test dataset size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1693)


# ## Training and Saving the Model

# Let's **train our model** and look at the model's features!

# In[14]:


# training the model for prediction
train_result = d4p.linear_regression_training().compute(X_train, y_train)


# To **get training model information** and **save it to a file**:

# In[17]:


# retrieving and printing training model
model = train_result.model
print("Here's our model:\n\n\n", model , "\n")

model_filename = './models/linear_regression_batch.pkl'

# saving model to a file
joblib.dump(model, model_filename) # nosec


# Now let's **load up the model** and look at one of the model's features.

# In[18]:


# loading the training model from a file
loaded_model = joblib.load(open(model_filename, "rb")) # nosec
print("Here is one of our loaded model's features: \n\n", loaded_model.Beta)


# ## Making a Prediction and Saving the Results

# Time to **make a prediction!**

# In[6]:


# now predicting the target feature(s) using the trained model
y_pred = d4p.linear_regression_prediction().compute(X_test, loaded_model).prediction 


# Now let's **export the results to a CSV file**.

# In[7]:


np.savetxt("./results/linear_regression_batch_results.csv", y_pred, delimiter =  ",")
print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")


# In[ ]:




