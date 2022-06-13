## Title
 Introduction to Machine Learning
  
## Requirements
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 20.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; AI Analytics Tookkit, Jupyter Notebooks, Intel DevCloud
|                                   | pip install seaborn
  
## Purpose
The Jupyter Notebooks in this training are inended to give professors and students an accesible but challenging introduction to machine learning.  It enumerates and describes many commonly used Scikit-learn* allgorithms which are used  daily to address machine learning challenges.  It has a secondary benefit of demonstrating how to accelerate commonly used Scikit-learn algorithms for Intel CPUs using Intel Extensions for Scikit-learn* which is part of the Intel AI Analytics Toolkit powered by oneAPI.

This workshop is designed to be used on the DevCloud and includes details on submitting batch jobs on the DevCloud environment.

## License  
Code samples 
are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.
Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Content Details

#### Pre-requisites

- Python* Programming
- Calculus
- Linear algebra
- Statistics


## Syllabus

- 11 Modules (18 hours)
- 11 Lab Exercises

-----------------------
| Modules | Description | Duration |
| :--- | :------ | :------ |
|[01_Introduction_to_Machine_Learning_and_Tools](01_Introduction_to_Machine_Learning_and_Tools/Introduction_to_Machine_Learning_and_Toolkit.ipynb)| + Demonstrate supervised learning algorithms.<br> + Explain key concepts like under- and over-fitting, regularization, and cross-validation<br>: + Classify the type of problem to be solved <br> + Choose an algorithm, tune parameters, and validate a model <br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 60 min |
|[02-Introduction_to_Supervised_Learning_and_K_Nearest_Neighbors](02-Introduction_to_Supervised_Learning_and_K_Nearest_Neighbors/Supervised_Learning_and_K_Nearest_Neighbors_Exercises.ipynb)| + Explain supervised learning and how it can be applied to regression and classification problems.<br> + Apply K-Nearest Neighbor (KNN) algorithm for classification <br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 120 min |
|[03-Train_Test_Splits_Validation_Linear_Regression](03-Train_Test_Splits_Validation_Linear_Regression/Train_Test_Splits_Validation_Linear_Regression.ipynb)| + Explain the difference between over-fitting and under-fitting a model<br> + Describe Bias-variance tradeoffs<br> + Find the optimal training and test data set splits, cross-validation, and model complexity versus error<br> + Apply a linear regression model for supervised learning.<br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 120 min |
|[04-Regularization_and_Gradient_Descent](04-Regularization_and_Gradient_Descent/Regularization_and_Gradient_Descent_Exercises.ipynb)| + Describe Logistic regression and how it differs from linear regression<br> + Identify metrics for classification errors and scenarios in which they can be used<br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 120 min |
|[05-Logistic_Regression_and_Classification_Error_Metrics](05-Logistic_Regression_and_Classification_Error_Metrics/Logistic_Regression_and_Classification_Error_Metrics_Exercises.ipynb)| + Explain cost functions, regularization, feature selection, and hyper-parameters<br> + | 120 min |
|[06-SVM_and_Kernels](06-SVM_and_Kernels/SVM_Kernels_Exercises.ipynb)| + Apply support vector machines (SVMs)—a popular algorithm used for classification problems<br> + Recognize SVM similarity to logistic regression<br> + Compute the cost function of SVMs<br> + Apply regularization in SVMs and some tips to obtain non-linear classifications with SVMs<nbr> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware | 120 min |
|[07-Decision_Trees](07-Decision_Trees/Decision_Trees_Exercises.ipynb)| + Recognize Decision trees and how to use them for classification problems<br> + Recognize how to identify the best split and the factors for splitting<br> + Explain strengths and weaknesses of decision trees<br> + Explain how regression trees help with classifying continuous values<br> + Describe motivation for choosing Random Forest Classifier over Decision Trees<br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware for Random Forest Classifier| 120 min |
|[08-Bagging](08-Bagging/Bagging_Exercises.ipynbb)| + Associate concepts of bootstrapping and aggregating (commonly known as “bagging”) to reduce variance<br> + Apply Random Forest algorithm that further reduces the correlation seen in bagging models<br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 120 min |
|[09-Boosting_and_Stacking](09-Boosting_and_Stacking/Boosting_and_Stacking_Exercises.ipynb)| + Explain how the boosting algorithm helps reduce variance and bias.<br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 120 min |
|[10-Introduction_to_Unsupervised_Learning_and_Clustering_Methods](10-Introduction_to_Unsupervised_Learning_and_Clustering_Methods/Clustering_Methods_Exercises.ipynb)| + Discuss unsupervised learning algorithms and how they can be applied<br> + Apply clustering<br> + Apply dimensionality reduction<br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 120 min |
|[11-Dimensionality_Reduction_and_Advanced_Topics](11-Dimensionality_Reduction_and_Advanced_Topics/Dimensionality_Reduction_Exercises.ipynb)| + Explain and Apply Principal Component Analysis (PCA)<br> + Explain Multidimensional Scaling (MDS)<br> + Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware| 120 min |
    
#### Content Structure

Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in Jupyter Lab to view the training contant, edit code and compile/run. 

## Install Directions

The training content can be accessed locally on the computer after installing necessary tools, or you can directly access using Intel DevCloud without any installation.

#### Local Installation of JupyterLab and oneAPI Tools

The Jupyter Notebooks can be downloaded locally to computer and accessed:
- Install Jupyter Lab on local computer: [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- Install Intel oneAPI Base Toolkit on local computer: [Installation Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) 
- git clone the repo and access the Notebooks using Jupyter Lab


#### Access using Intel DevCloud

The Jupyter notebooks are tested and can be run on Intel DevCloud without any installation necessary, below are the steps to access these Jupyter notebooks on Intel DevCloud:
1. Register on [Intel DevCloud](https://devcloud.intel.com/oneapi)
2. Login, Get Started and Launch Jupyter Lab
3. Open Terminal in Jupyter Lab and git clone the repo and access the Notebooks
