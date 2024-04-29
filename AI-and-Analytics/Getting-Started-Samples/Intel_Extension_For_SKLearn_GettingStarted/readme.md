# Intel® Python Scikit-learn Extension Getting Started Sample

The `Intel® Python Scikit-learn Extension Getting Started` sample demonstrates how to use a support vector machine classifier from Intel® Extension for Scikit-learn* for digit recognition problem. All other machine learning algorithms available with Scikit-learn can be used in the similar way. Intel® Extension for Scikit-learn* speeds up scikit-learn applications. The acceleration is achieved through the use of the Intel® oneAPI Data Analytics Library (oneDAL) [Intel oneAPI Data Analytics Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html), which comes with [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).


| Area                     | Description
|:---                      | :---
| Category                 | Getting Started
| What you will learn      | How to use a basic Intel® Extension for Scikit-learn* programming model for Intel CPUs
| Time to complete         | 5 minutes


## Purpose

In this sample, you will run a support vector classifier model from sklearn with oneDAL Daal4py library memory objects. You will also learn how to train a model and save the information to a file. Intel® Extension for Scikit-learn* depends on Intel® Daal4py. Daal4py is a simplified API to oneDAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users. Built to help provide an abstraction to oneDAL for direct usage or integration into one's own framework.
## Prerequisites

| Optimized for            | Description
| :---                     | :---
| OS                       | Ubuntu* 20.04 (or newer)
| Hardware                 | Intel Atom® processors <br> Intel® Core™ processor family  <br> Intel® Xeon® processor family  <br> Intel® Xeon® Scalable processor family
| Software                 | Intel® AI Analytics Toolkit (AI Kit) <br> Intel® oneAPI Data Analytics Library (oneDAL) <br> Joblib  <br> NumPy <br> Matplotlib

You can refer to the oneAPI [product page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit *[Get Started with the Intel® AI Analytics Toolkit for Linux*
](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit)* for post-installation steps and scripts.

oneDAL is ready for use once you finish the AI Kit installation and have run the post installation script.


## Key Implementation Details

This Getting Started sample code is implemented for CPU using the Python language. The example assumes you have Intel® Extension for Scikit-learn* installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the [Intel® AI Analytics Toolkit](https://software.intel.com/en-us/oneapi/ai-kit). Intel® Extension for Scikit-learn* is available as a part of Intel® AI Analytics Toolkit (AI kit).

## Environment Setup

1. If you have not already done so, activate the AI Tools bundle base environment. If you used the default location to install AI Tools, open a terminal and type the following
```
source $HOME/intel/oneapi/intelpython/bin/activate
```
If you used a separate location, open a terminal and type the following
```
source <custom_path>/bin/activate
```

2. Activate Conda with Root Access

Intel Python environment will be active by default. However, if you activated another environment, you can return with the following command.
```
source activate base
pip install -r requirements.txt
```

2a. Activate Conda without Root Access (Optional)

By default, the Intel® AI Analytics Toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone and activate your desired conda environment using the following commands.
```
conda create --name usr_intelpython --clone base
source activate usr_intelpython
```
3. Clone the GitHub repository
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples
```

### Install Jupyter Notebook

1. Change to the sample directory.
2. Install Jupyter Notebook with the proper kernel.
   ```
   conda install jupyter nb_conda_kernels
   ```

#### View in Jupyter Notebook

>**Note**: This distributed execution cannot be launched from Jupyter Notebook, but you can still view inside the notebook to follow the included write-up and description.

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook
   ```
3. Locate and select the Notebook.
   ```
   Intel_Extension_For_SKLearn_GettingStarted.ipynb
   ```
4. Click the **Run** button to move through the cells in sequence.


#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

You should see printed output for cells (with similar numbers) and an accuracy result.

![](images/sample_digit_images.JPG "Image samples from dataset")

![](images/predicted.JPG "Predicted digits for random test images")

```
Model accuracy on test data: 0.9833333333333333

[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```

## Related Samples

* [Intel® Python XGBoost* Getting Started](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/IntelPython_XGBoost_GettingStarted)
* [Intel® Python XGBoost Daal4py Prediction](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPython_XGBoost_daal4pyPrediction)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
