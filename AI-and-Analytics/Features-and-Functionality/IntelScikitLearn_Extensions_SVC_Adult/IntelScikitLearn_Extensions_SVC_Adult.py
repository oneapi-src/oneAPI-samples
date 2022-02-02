#!/usr/bin/env python
# coding: utf-8

# Intel® Extension for Scikit-learn: SVC for Adult dataset

from time import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

x, y = fetch_openml(name='a9a', return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Intel(R) Extension for Scikit-learn (previously known as daal4py) contains drop-in replacement
# functionality for the stock scikit-learn package. You can take advantage of the performance
# optimizations of Intel® Extension for Scikit-learn by adding just two lines of code before the
# usual scikit-learn imports:

from sklearnex import patch_sklearn
patch_sklearn()

# Intel(R) Extension for Scikit-learn patching affects performance of specific
# Scikit-learn functionality. Refer to the list of supported algorithms and parameters:
# https://intel.github.io/scikit-learn-intelex/algorithms.html for details.
# In cases when unsupported parameters are used, the package fallbacks into original Scikit-learn.

params = {
    'C': 100.0,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# "Train SVC algorithm with Intel(R) Extension for Scikit-learn on Adult dataset:"

start = time()
from sklearn.svm import SVC
classifier = SVC(**params).fit(x_train, y_train)
print(f"Execution time with Intel(R) Extension for Scikit-learn: {(time() - start):.2f} s")

# Make predictions with SVC classifier and print a report of the main classification metrics:

predicted = classifier.predict(x_test)
report = metrics.classification_report(y_test, predicted)
print(f"Classification report for SVC trained with Intel(R) extension for Scikit-learn:\n{report}\n")

# To cancel optimizations, use *unpatch_sklearn* and reimport the SVC class.

from sklearnex import unpatch_sklearn
unpatch_sklearn()

# "Train SVC algorithm with original scikit-learn:"

start = time()
from sklearn.svm import SVC
classifier = SVC(**params).fit(x_train, y_train)
print(f"Execution time with the original Scikit-learn: {(time() - start):.2f} s")

# Make predictions with SVC classifier and print a report of the main classification metrics:

predicted = classifier.predict(x_test)
report = metrics.classification_report(y_test, predicted)
print(f"Classification report for SVC trained with the original scikit-learn:\n{report}\n")

# With Intel(R) Extension for Scikit-learn you can:
# 
# - Use your existing scikit-learn code for training and prediction;
# - Add a couple of lines to execute your code up to **72** times faster;
# - Get models of the same quality.
