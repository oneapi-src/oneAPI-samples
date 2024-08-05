# Intel® Python Daal4py Getting Started Sample

The `Intel® Python Daal4py Getting Started` sample code shows how to do batch linear regression using the Python API package daal4py powered by the [Intel® oneAPI Data Analytics Library (oneDAL)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html).

| Area                   | Description
| :---                   | :---
| Category               | Getting Started
| What you will learn    | Basic daal4py programming model for Intel CPUs
| Time to complete       | 5 minutes


## Purpose

daal4py is a simplified API to oneDAL that allows for fast usage of the framework suited for data scientists or machine learning users. Built to help provide an abstraction to Intel® oneDAL for direct usage or integration into one's own framework.

In this sample, you will run a batch Linear Regression model with oneDAL daal4py library memory objects. You will also learn how to train a model and save the information to a file.

## Prerequisites

| Optimized for           | Description
| :---                    | :---
| OS                      | Ubuntu* 20.04 (or newer)
| Hardware                | Intel Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable processor family
| Software                | Intel® oneAPI Data Analytics Library (oneDAL)
> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Tools**

  You can get the AI Tools from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Tools for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.


## Key Implementation Details

- This get started sample code is implemented for CPUs using the Python language. The example assumes you have daal4py and scikit-learn installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the Intel® AI Analytics Toolkit.

- The Intel® oneAPI Data Analytics Library (oneDAL) is ready for use once you finish the Intel® AI Analytics Toolkit installation and have run the post installation script.

## Environment Setup

You will need to download and install the following toolkits, tools, and components to use the sample.

**1. Get Intel® AI Tools**

Required AI Tools: Intel® Optimization for XGBoost*
<br>If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

**2. Install dependencies**
```
pip install -r requirements.txt
```
**Install Jupyter Notebook** by running `pip install notebook`. Alternatively, see [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.

## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/IntelPython_daal4py_GettingStarted#environment-setup) is completed.
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
<!-- specify the name of conda environment for this sample, if there are several options - list all relevant names of environments -->
```
conda activate xgboost
``` 
3. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/IntelPython_daal4py_GettingStarted
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
IntelPython_daal4py_GettingStarted.ipynb
```

7. Change the kernel to xgboost
 
8. Run every cell in the Notebook in sequence.

### Conda/PIP
> **Note**: Make sure your Conda/Python environment with AI Tools installed is activated
1. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneapi-samples/AI-and-Analytics/Getting-Started-Samples/IntelPython_daal4py_GettingStarted
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
IntelPython_daal4py_GettingStarted.ipynb.ipynb
```

6. Run every cell in the Notebook in sequence.

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.



## Example Output

```
Here's our model:

 NumberOfBetas: 14

NumberOfResponses: 1

InterceptFlag: False

Beta: array(
  [[ 0.00000000e+00 -1.05416344e-01  5.25259886e-02  4.26844883e-03
     2.76607367e+00 -2.82517989e+00  5.49968304e+00  3.48833264e-03
    -8.73247684e-01  1.74005447e-01 -8.38917510e-03 -3.28044397e-01
     1.58423529e-02 -4.57542900e-01]],
  dtype=float64, shape=(1, 14))

NumberOfFeatures: 13

Here is one of our loaded model's features:

 [[ 0.00000000e+00 -1.05416344e-01  5.25259886e-02  4.26844883e-03
   2.76607367e+00 -2.82517989e+00  5.49968304e+00  3.48833264e-03
  -8.73247684e-01  1.74005447e-01 -8.38917510e-03 -3.28044397e-01
   1.58423529e-02 -4.57542900e-01]]
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```
## Related Samples

* [Intel® Python XGBoost* Getting Started Sample](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/IntelPython_XGBoost_GettingStarted)
* [Intel® Python Scikit-learn Extension Getting Started Sample](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_SKLearn_GettingStarted#intel-python-scikit-learn-extension-getting-started-sample)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
