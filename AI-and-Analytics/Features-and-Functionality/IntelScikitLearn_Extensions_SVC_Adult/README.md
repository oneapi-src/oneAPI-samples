# `Intel® Extension for Scikit-learn: SVC for Adult dataset` Sample
This sample code uses [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) to show how to train and predict with a SVC algorithm using Intel® Extension for Scikit-learn. It demonstrates how to use software products that can be found in the [Intel(R) oneAPI Data Analytics Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html), [Intel(R) Extension for Scikit-learn](https://intel.github.io/scikit-learn-intelex/), or [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher, 64-bit Windows 10, macOS 10.14 or higher
| Hardware                          | Intel Atom® Processors; Intel® Core™ Processor Family; Intel® Xeon® Processor Family; Intel® Xeon® Scalable Performance Processor Family
| Software                          | Intel® oneAPI AI Analytics Toolkit
| What you will learn               | How to get started with Intel(R) Extension for Scikit-learn
| Time to complete                  | 25 minutes

## Purpose

Intel® Extension for Scikit-learn is a seamless way to speed up your Scikit-learn application. The acceleration is achieved through the use of the Intel(R) oneAPI Data Analytics Library ([oneDAL](https://github.com/oneapi-src/oneDAL)). Patching scikit-learn makes it a well-suited machine learning framework for dealing with real-life problems.

In this sample, you will run a SVC algorithm with Intel® Extension for Scikit-learn. You will see a significant increase in performance over the original Scikit-learn while maintaining the same precision.
  
## Key Implementation Details 
The sample code is written in Python and it targets CPU architecture. The example assumes you have Intel® Extension for Scikit-learn installed.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Running Samples on the Intel&reg; DevCloud
If you are running this sample on the DevCloud, see [Running Samples on the Intel&reg; DevCloud](#run-samples-on-devcloud)

## Build and Run the Sample

### Pre-requirement

> NOTE: No action is required if you are using Intel DevCloud as your environment.
  Refer to [Intel oneAPI DevCloud](https://intelsoftwaresites.secure.force.com/devcloud/oneapi) for Intel DevCloud.

 1. **Intel® AI Analytics Toolkit**
       You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation,
       and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

 2. **Jupyter Notebook**
       Users can install via PIP by `$pip install notebook`.
       Users can also refer to the [installation link](https://jupyter.org/install) for details.


### Running the Sample as a Jupyter Notebook

1. Launch Jupyter notebook: `$jupyter notebook --ip=0.0.0.0`
2. Follow the instructions to open the URL with the token in your browser
3. Click the `IntelScikitLearn_Extensions_SVC_Adult.ipynb` file
4. Run through every cell of the notebook one by one

### Running the Sample as a Python File

1. python `IntelScikitLearn_Extensions_SVC_Adult.py`

### Example of Output

```
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
Intel(R) extension for Scikit-learn time: 13.88 s
Classification report for SVC:
              precision    recall  f1-score   support

        -1.0       0.87      0.90      0.88      7414
         1.0       0.64      0.58      0.61      2355

    accuracy                           0.82      9769
   macro avg       0.76      0.74      0.75      9769
weighted avg       0.82      0.82      0.82      9769


Original Scikit-learn time: 1019.58 s
Classification report for SVC:
              precision    recall  f1-score   support

        -1.0       0.87      0.90      0.88      7414
         1.0       0.64      0.58      0.61      2355

    accuracy                           0.82      9769
   macro avg       0.76      0.74      0.75      9769
weighted avg       0.82      0.82      0.82      9769
```