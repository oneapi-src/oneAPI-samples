# `Intel® Extension for Scikit-learn: SVC for Adult dataset` Sample
This sample code uses [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) to show how to train and predict with a SVC algorithm using Intel® Extension for Scikit-learn. It demonstrates how to use software products that can be found in the [Intel® oneAPI Data Analytics Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html), [Intel(R) Extension for Scikit-learn](https://intel.github.io/scikit-learn-intelex/), or [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher, 64-bit Windows 10, macOS 10.14 or higher
| Hardware                          | Intel Atom® Processors; Intel® Core™ Processor Family; Intel® Xeon® Processor Family; Intel® Xeon® Scalable processor family
| Software                          | Intel® oneAPI AI Analytics Toolkit
| What you will learn               | How to get started with Intel® Extension for Scikit-learn
| Time to complete                  | 25 minutes

## Purpose

Intel® Extension for Scikit-learn is a seamless way to speed up your Scikit-learn application. The acceleration is achieved through the use of the Intel® oneAPI Data Analytics Library ([oneAPI Data Analytics Library (oneDAL)](https://github.com/oneapi-src/oneDAL)). Patching scikit-learn makes it a well-suited machine learning framework for dealing with real-life problems.

In this sample, you will run a SVC algorithm with Intel® Extension for Scikit-learn and compare its performance against the original stock version of scikit-learn. You will see that patching scikit-learn results in a significant increase in performance over the original scikit-learn while also maintaining the same precision.

## Key Implementation Details
The sample code is written in Python and it targets CPU architecture. The example assumes you have Intel® Extension for Scikit-learn installed.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run the Sample

### Pre-requirement

> NOTE: No action is required if you are using Intel DevCloud as your environment.
  Refer to [Intel® DevCloud for oneAPI](https://intelsoftwaresites.secure.force.com/devcloud/oneapi) for Intel DevCloud.

 1. **Intel® AI Analytics Toolkit**
       Install the toolkit from the [oneAPI main page](https://software.intel.com/en-us/oneapi)
	     and refer to the [Toolkit Get Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

 2. **Jupyter Notebook**
       Install Jupyter Notebook via pip: `pip install notebook`.
       Refer to [Installing the Jupyter Software](https://jupyter.org/install) for details.


### Running the Sample as a Jupyter Notebook

1. Launch Jupyter notebook: `jupyter notebook --ip=0.0.0.0`
2. Follow the instructions to open the URL with the token in your browser.
3. Click the `IntelScikitLearn_Extensions_SVC_Adult.ipynb` file.
4. Run each cell of the notebook one by one.

### Running the Sample as a Python File

1. `IntelScikitLearn_Extensions_SVC_Adult.py`

### Example of Output

```
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
Execution time with Intel(R) extension for Scikit-learn: 13.88 s
Classification report for SVC trained with Intel(R) extension for Scikit-learn:
              precision    recall  f1-score   support

        -1.0       0.87      0.90      0.88      7414
         1.0       0.64      0.58      0.61      2355

    accuracy                           0.82      9769
   macro avg       0.76      0.74      0.75      9769
weighted avg       0.82      0.82      0.82      9769


Execution time with the original Scikit-learn: 1019.58 s
Classification report for SVC trained with the original scikit-learn:
              precision    recall  f1-score   support

        -1.0       0.87      0.90      0.88      7414
         1.0       0.64      0.58      0.61      2355

    accuracy                           0.82      9769
   macro avg       0.76      0.74      0.75      9769
weighted avg       0.82      0.82      0.82      9769
```
### Using Visual Studio Code*  (VS Code)

You can use VS Code extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.