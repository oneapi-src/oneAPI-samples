#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
=============================================================
Copyright © 2020 Intel Corporation
 
SPDX-License-Identifier: MIT
=============================================================
'''

# # Daal4py Linear Regression Example for Distributed Memory Systems [SPMD mode]

# ## IMPORTANT NOTICE
# 

# When using daal4py for distributed memory systems, the command needed to execute the program should be **executed 
# in a bash shell**. In order to run this example, please download it as a .py file then run the following command (**the number 4 means that it will run on 4 processes**):

# mpirun -n 4 python ./daal4py_Distributed_LinearRegression.py

# ## Importing and Organizing Data

# In this example we will be predicting **prices of houses in Boston** based on the features of each house.
# 
# Let's start by **importing** all necessary data and packages.

# In[2]:


##### daal4py linear regression example for distributed memory systems [SPMD mode] #####
import daal4py as d4p
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle


# Now let's **load** in the dataset and **organize** it as necessary to work with our model. For distributed, every file has a unique ID.
# 
# We will also **initialize the distribution engine**.

# In[3]:


d4p.daalinit() #initializes the distribution engine

# organizing variables used in the model for prediction
# each process gets its own data
infile = "./data/distributed_data/linear_regression_train_" + str(d4p.my_procid()+1) + ".csv"

# read data
indep_data = pd.read_csv(infile).drop(["target"], axis=1) # house characteristics
dep_data   = pd.read_csv(infile)["target"] # house price


# ## Training and Saving the Model

# Time to **train our model** and look at the model's features! 

# In[4]:


# training the model for prediction
train_result = d4p.linear_regression_training(distributed=True).compute(indep_data, dep_data)


# To **get training model information** and **save it to a file**:

# In[5]:


# retrieving and printing training model
model = train_result.model
print("Here's our model:\n\n\n",model , "\n")

model_filename = './models/daal4py_Distributed_LinearRegression_' + str(d4p.my_procid()+1) + '.sav'

# saving model to a file
pickle.dump(model, open(model_filename, "wb"))


# Now let's **load up the model** and look at one of the model's features.

# In[6]:


# loading the training model from a file
loaded_model = pickle.load(open(model_filename, "rb"))
print("Here is one of our loaded model's features: \n\n",loaded_model.Beta)


# ## Making a Prediction and Saving the Results

# Time to **make a prediction!**

# In[9]:


# read test data
test_data = pd.read_csv("./data/distributed_data/linear_regression_test.csv").drop(["target"], axis=1)

# now predict using the model from the training above
predict_result = d4p.linear_regression_prediction().compute(test_data, train_result.model).prediction


# Now let's **export the results to a CSV file**. We will also **stop the distribution engine.**

# In[10]:


# now export the results to a CSV file
results_filename = "./results/daal4py_Distributed_LinearRegression_results" + str(d4p.my_procid()+1) + ".csv"
np.savetxt(results_filename, predict_result, delimiter =  ",")

d4p.daalfini() # stops the distribution engine
print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')

