#!/usr/bin/env python
# coding: utf-8

# Intel® Extension for Scikit-learn SVC for Adult dataset

from time import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

x, y = fetch_openml(name='a9a', return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Intel Extension for Scikit-learn (previously known as daal4py) contains drop-in replacement
# functionality for the stock scikit-learn package. You can take advantage of the performance
# optimizations of Intel Extension for Scikit-learn by adding just two lines of code before the
# usual scikit-learn imports:

from sklearnex import patch_sklearn
patch_sklearn()

# Intel(R) Extension for Scikit-learn patching affects performance of specific
# Scikit-learn functionality. Refer to the list of supported algorithms and parameters:
# https://intel.github.io/scikit-learn-intelex/algorithms.html for details.
# In cases when unsupported parameters are used, the package fallbacks into original Scikit-learn

params = {
    'C': 100.0,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# Training of the SVC algorithm with Intel(R) Extension for Scikit-learn for Adult dataset

start = time()
from sklearn.svm import SVC
classifier = SVC(**params).fit(x_train, y_train)
print(f"Intel(R) extension for Scikit-learn time: {(time() - start):.2f} s")

# Predict and get a result of the SVC algorithm with Intel(R) Extension for Scikit-learn

predicted = classifier.predict(x_test)
report = metrics.classification_report(y_test, predicted)
print(f"Classification report for SVC:\n{report}\n")

# In order to cancel optimizations, we use *unpatch_sklearn* and reimport the class SVC.

from sklearnex import unpatch_sklearn
unpatch_sklearn()

# Training of the SVC algorithm with original scikit-learn library for Adult dataset

start = time()
from sklearn.svm import SVC
classifier = SVC(**params).fit(x_train, y_train)
print(f"Original Scikit-learn time: {(time() - start):.2f} s")

# Predict and get a result of the SVC algorithm with original Scikit-learn

predicted = classifier.predict(x_test)
report = metrics.classification_report(y_test, predicted)
print(f"Classification report for SVC:\n{report}\n")

# With scikit-learn-intelex patching you can:
# 
# - Use your scikit-learn code for training and prediction with minimal changes (a couple of lines of code);
# - Fast execution training and prediction of scikit-learn models;
# - Get the same quality;
# - Get speedup more than **72** times.
