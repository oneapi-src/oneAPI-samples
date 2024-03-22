# Intel® Python Scikit-learn Extension Getting Started Sample

The `Intel® Python Scikit-learn Extension Getting Started` sample demonstrates how to use a support vector machine classifier from Intel® Extension for Scikit-learn* for digit recognition problem. All other machine learning algorithms available with Scikit-learn can be used in the similar way. Intel® Extension for Scikit-learn* speeds up scikit-learn applications. The acceleration is achieved through the use of the Intel® oneAPI Data Analytics Library (oneDAL) [Intel oneAPI Data Analytics Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html).


| Area                     | Description
|:---                      | :---
| Category                 | Getting Started
| What you will learn      | How to use a basic Intel® Extension for Scikit-learn* programming model for Intel CPUs
| Time to complete         | 5 minutes

## Prerequisites

| Optimized for            | Description
| :---                     | :---
| OS                       | Ubuntu* 20.04 (or newer)
| Hardware                 | Intel Atom® processors <br> Intel® Core™ processor family  <br> Intel® Xeon® processor family  <br> Intel® Xeon® Scalable processor family
| Software                 | Intel® AI Analytics Toolkit (AI Kit) <br> Intel® oneAPI Data Analytics Library (oneDAL) <br> Joblib  <br> NumPy <br> Matplotlib

You can refer to the oneAPI [product page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit *[Get Started with the Intel® AI Analytics Toolkit for Linux*
](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit)* for post-installation steps and scripts.

oneDAL is ready for use once you finish the AI Kit installation and have run the post installation script.

## Purpose

In this sample, you will run a support vector classifier model from sklearn with oneDAL Daal4py library memory objects. You will also learn how to train a model and save the information to a file. Intel® Extension for Scikit-learn* depends on Intel® Daal4py. Daal4py is a simplified API to oneDAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users. Built to help provide an abstraction to oneDAL for direct usage or integration into one's own framework.

## Key Implementation Details

This Getting Started sample code is implemented for CPU using the Python language. Intel® Extension for Scikit-learn* is available as a part of Intel® AI Tools.

You will need to download and install the following toolkits, tools, and components to use the sample.

**1. Get Intel® AI Tools**

Required AI Tools: Intel® Extension for Scikit-learn*
<br>If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

**2. Install dependencies**
```
pip install -r requirements.txt
```
**Install Jupyter Notebook** by running `pip install notebook`. Alternatively, see [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.

## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_SKLearn_GettingStarted#environment-setup) is completed.
Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip) 
* [Docker](#docker)

### AI Tools Offline Installer (Validated)  
1. If you have not already done so, activate the AI Tools bundle base environment. If you used the default location to install AI Tools, open a terminal and type the following
```
source $HOME/intel/oneapi/intelpython/bin/activate
```
If you used a separate location, open a terminal and type the following
```
source <custom_path>/bin/activate
```
2. Activate the Conda environment:
```
conda activate sklearnex
``` 
3. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneapi-samples/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_SKLearn_GettingStarted
```

4. Launch Jupyter Notebook: 
> **Note**: You might need to register Conda kernel to Jupyter Notebook kernel, 
feel free to check [the instruction](https://github.com/IntelAI/models/tree/master/docs/notebooks/perf_analysis#option-1-conda-environment-creation)
```
jupyter notebook --ip=0.0.0.0
```
<!-- add other flags to jupyter notebook command if needed, such as port 8888 or allow-root -->
5. Follow the instructions to open the URL with the token in your browser.
6. Select the Notebook:
```
Intel_Extension_For_SKLearn_GettingStarted.ipynb
```
7. Change the kernel to sklearnex
  
8. Run every cell in the Notebook in sequence.

### Conda/PIP
> **Note**: Make sure your Conda/Python environment with AI Tools installed is activated
1. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneapi-samples/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_SKLearn_GettingStarted
```
2. Launch Jupyter Notebook: 
> **Note**: You might need to register Conda kernel to Jupyter Notebook kernel, 
feel free to check [the instruction](https://github.com/IntelAI/models/tree/master/docs/notebooks/perf_analysis#option-1-conda-environment-creation)
```
jupyter notebook --ip=0.0.0.0
```
<!-- add other flags to jupyter notebook command if needed, such as port 8888 or allow-root -->
4. Follow the instructions to open the URL with the token in your browser.
5. Select the Notebook:
```
Intel_Extension_For_SKLearn_GettingStarted.ipynb
```

6. Run every cell in the Notebook in sequence.

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

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
* [Intel® Python Daal4py Getting Started](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/IntelPython_daal4py_GettingStarted)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
