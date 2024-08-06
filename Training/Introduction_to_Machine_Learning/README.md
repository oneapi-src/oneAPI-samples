# Introduction to Machine Learning

The Jupyter Notebooks in these samples are intended to give professors and students an accessible but challenging introduction to machine learning. It enumerates and describes many commonly used Scikit-learn* algorithms which are used daily to address machine learning challenges. The collection has a secondary benefit of demonstrating how to accelerate commonly used Scikit-learn algorithms for Intel CPUs using Intel® Extension for Scikit-learn* which is part of the Intel® AI Analytics Toolkit (AI Kit). This workshop is designed to be used on the DevCloud and includes details on submitting batch jobs on the DevCloud environment.

This workshop is designed to be used on Intel® DevCloud and includes details on submitting batch jobs on the Intel® DevCloud environment.

## Prerequisites

| Optimized for             | Description
|:---                       |:---
| OS                        | Ubuntu* 20.04 (or newer) <br> Windows* 10
| Hardware                  | Gen (or newer)
| Software                  | Intel® oneAPI Base Toolkit (Base Kit) <br> Intel® AI Analytics Toolkit (AI Kit) <br> pip install seaborn

Additionally, you will need to know about:

- Python* programming
- Calculus
- Linear algebra
- Statistics

## Jupyter Notebook and Descriptions

|Modules                               |Description                  |Recommended Video            |Duration
|:---                                  |:---                         |:---                         |:---
|[Introduction to Machine Learning and Tools](01_Introduction_to_Machine_Learning_and_Tools/Introduction_to_Machine_Learning_and_Toolkit.ipynb) | - Classify the type of problem to be solved <br> - Demonstrate supervised learning algorithms <br> - Choose an algorithm, tune parameters, and validate a model <br> - Explain key concepts like under- and over-fitting, regularization, and cross-validation <br> - Apply Intel Extension for Scikit-learn* patching to leverage underlying compute capabilities of hardware. |Introduction to Intel® Extension for Scikit-learn* | 60 min
|[Supervised Learning and K Nearest Neighbors](02-Supervised_Learning_and_K_Nearest_Neighbors/Supervised_Learning_and_K_Nearest_Neighbors_Exercises.ipynb) | - Explain supervised learning as applied to regression and classification problems <br> - Apply K-Nearest Neighbor (KNN) algorithm for classification <br> - Apply patching to leverage underlying compute capabilities of hardware | KNearest Neighbor | 120 min
|[Train Test Splits Validation Linear Regression](03-Train_Test_Splits_Validation_Linear_Regression/Train_Test_Splits_Validation_Linear_Regression.ipynb) | - Explain the difference between over-fitting and under-fitting <br> - Describe Bias-variance tradeoffs <br> - Find the optimal training and test data set splits <br> - Apply cross-validation <br> - Apply a linear regression model for supervised learning <br> - Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware | Introduction to Intel® Extension for Scikit-learn* | 120 min
|[Regularization and Gradient Descent](04-Regularization_and_Gradient_Descent/Regularization_and_Gradient_Descent_Exercises.ipynb) | - Explain cost functions, regularization, feature selection, and hyper-parameters <br> - Summarize complex statistical optimization algorithms like gradient descent and its application to linear regression <br> - Apply patching to leverage underlying compute capabilities of hardware | N/A | 120 min
|[Logistic Regression and Classification Error Metrics](05-Logistic_Regression_and_Classification_Error_Metrics/Logistic_Regression_and_Classification_Error_Metrics_Exercises.ipynb) | - Describe Logistic regression and how it differs from linear regression <br> - Identify metrics for classification errors and scenarios in which they can be used <br> - Apply patching to leverage underlying compute capabilities of hardware | Logistic Regression Walkthrough | 120 min
|[SVM and Kernels](06-SVM_and_Kernels/SVM_Kernels_Exercises.ipynb) | - Apply support vector machines (SVMs) for classification problems <br> - Recognize SVM similarity to logistic regression <br> - Compute the cost function of SVMs <br> - Apply regularization in SVMs and some tips to obtain non-linear classifications with SVMs <br> - Apply patching to leverage underlying compute capabilities of hardware | N/A | 120 min
|[Decision Trees](07-Decision_Trees/Decision_Trees_Exercises.ipynb) | - Recognize Decision trees and apply them for classification problems <br> - Recognize how to identify the best split and the factors for splitting <br> - Explain strengths and weaknesses of decision trees <br> - Explain how regression trees help with classifying continuous values <br> - Describe motivation for choosing Random Forest Classifier over Decision Trees <br> - Apply patching to Random Forest Classifier | N/A | 120 min
|[Bagging](08-Bagging/Bagging_Exercises.ipynb) | - Describe bootstrapping and aggregating (a.k.a. “bagging”) to reduce variance <br> - Reduce the correlation seen in bagging using Random Forest algorithm <br> - Apply patching to leverage underlying compute capabilities of hardware | N/A | 120 min
|[Boosting and Stacking](09-Boosting_and_Stacking/Boosting_and_Stacking_Exercises.ipynb) | - Explain how the boosting algorithm helps reduce variance and bias <br> - Apply patching to leverage underlying compute capabilities of hardware | N/A | 120 min
|[Introduction to Unsupervised Learning and Clustering Methods](10-Introduction_Clustering_Methods/Clustering_Methods_Exercises.ipynb) | - Describe unsupervised learning algorithms their application <br> - Apply clustering <br> - Apply dimensionality reduction <br> - Apply patching to leverage underlying compute capabilities of hardware | KMeans Walkthrough <br> Introduction to Intel® Extension for Scikit-learn* | 120 min
|[Dimensionality Reduction and Advanced Topics](11-Dimensionality_Reduction_and_Advanced_Topics/Dimensionality_Reduction_Exercises.ipynb) | - Explain and Apply Principal Component Analysis (PCA) <br> - Explain Multidimensional Scaling (MDS) <br> - Apply patching to leverage underlying compute capabilities of hardware | PCA Walkthrough | 120 min

### Content Structure

Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in JupyterLab to view the training content, edit code, and compile and run.

The training content can be accessed locally on the computer after installing necessary tools, or you can directly access the same material using Intel® DevCloud, which does not require separate installation.

## Run the Jupyter Notebooks Locally (on Linux* or WSL)

1. Update the package manager on your system.
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. Install JupyterLab. See the [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) for more information. 

3. Download and install Intel® oneAPI Base Toolkit (Base Kit) and Intel® AI Analytics Toolkit (AI Kit) from the [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html) page.

4. After you complete the installation, refresh the new environment variables.
   ```bash
   source .bashrc
   ```

5. Initialize the oneAPI environment enter.
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

6. Clone the oneAPI-samples GitHub repository.

   >**Note**: If Git is not installed, install it now.
   >```bash
   >sudo apt install git
   >```

   ```bash
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```

7. From a terminal, start JupyterLab.
    ```bash
    jupyter lab
    ```

8. Make note of the address printed in the terminal, and paste the address into your browser address bar.

9. From the navigation panel, navigate through the directory structure and select a Notebook to run. (The notebooks have a `.ipynb ` extension.)


## Run the Jupyter Notebooks on Intel® Devcloud

Use these general steps to access the notebooks on the Intel® Devcloud for oneAPI.

>**Note**: For more information on using Intel® DevCloud, see the Intel® oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started/) page.

1. If you do not already have an account, request an Intel® DevCloud account at [Create an Intel® DevCloud Account](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).

2. Once you get your credentials, log in to the Intel® DevCloud using [JupyterLab](https://jupyter.oneapi.devcloud.intel.com/hub/login?next=/lab/tree/Welcome.ipynb?reset) to connect with your account credentials.

3. Open a terminal, and clone the GitHub repository into your account.
   ```bash
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```
4. From the navigation panel, navigate through the directory structure and select a Notebook to run. (The notebooks have a `.ipynb ` extension.)

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).