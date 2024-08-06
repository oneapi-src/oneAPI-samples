## Title
 oneAPI Essentials
  
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

At the end of this course you will be able to:

- Explain the difference between over-fitting and underfitting a model
- Describe Bias-variance tradeoffs
- Find the optimal training and test data set splits, crossvalidation, and model complexity versus error
- Apply a linear regression model for supervised learning
- Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware using two lines of code
``` python
      from sklearnex import patch_sklearn
      patch_sklearn()
```

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
