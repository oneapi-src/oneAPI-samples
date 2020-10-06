#!/usr/bin/env python
# coding: utf-8

# In[2]:


# =============================================================
# Copyright © 2020 Intel Corporation
# 
# SPDX-License-Identifier: MIT
# =============================================================


# # XGBoost Getting Started Example on Linear Regression
# ## Importing and Organizing Data
# In this example we will be predicting prices of houses in Boston based on the features of each house using Intel optimized XGBoost shipped as a part of the oneAPI AI Analytics Toolkit.
# Let's start by **importing** all necessary data and packages.

# In[3]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Now let's **load** in the dataset and **organize** it as necessary to work with our model.

# In[4]:


#loading the data
boston = load_boston()

#converting data into a pandas dataframe
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

#setting price as value to be predicted
data['PRICE'] = boston.target

#extracting rows
X, y = data.iloc[:,:-1],data.iloc[:,-1]

#using dmatrix values for xgboost
data_dmatrix = xgb.DMatrix(data=X,label=y)

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1693)


# **Instantiate and define XGBoost regresion object** by calling the XGBRegressor() class from the library. Use hyperparameters to define the object.

# In[5]:


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)


# ## Training and Saving the model

# **Fitting and training model** using training datasets and predicting values.

# In[6]:


xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)


# **Finding root mean squared error** of predicted values.

# In[7]:


rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:",rmse)


#  ##Saving the Results

# Now let's **export the predicted values to a CSV file**.

# In[8]:


pd.DataFrame(preds).to_csv('foo.csv',index=False)


# In[9]:


print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")


# In[ ]:




