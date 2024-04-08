# Intel® Python XGBoost* Getting Started Sample

The `Intel® Python XGBoost* Getting Started` sample demonstrates how to set up and train an XGBoost model on datasets for prediction.

| Area                     | Description
| :---                     | :---
| Category                 | Getting Started
| What you will learn      | The basics of XGBoost programming model for Intel CPUs
| Time to complete         | 5 minutes

## Purpose

XGBoost* is a widely used gradient boosting library in the classical ML area. Designed for flexibility, performance, and portability, XGBoost* includes optimized distributed gradient boosting frameworks and implements Machine Learning algorithms underneath. Starting with 0.9 version of XGBoost, Intel has been up streaming optimizations through the `hist` histogram tree-building method. Starting with 1.3.3 version of XGBoost and beyond, Intel has also begun up streaming inference optimizations to XGBoost as well.

In this code sample, you will learn how to use Intel optimizations for XGBoost published as part of Intel® AI Tools. The sample also illustrates how to set up and train an XGBoost* model on datasets for prediction. It also demonstrates how to use software products that can be found in the [Intel® AI Tools](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

## Prerequisites

| Optimized for           | Description
| :---                    | :---
| OS                      | Ubuntu* 20.04 (or newer)
| Hardware                | Intel Atom® Processors <br> Intel® Core™ Processor Family <br> Intel® Xeon® Processor Family <br> Intel® Xeon® Scalable processor family
| Software                | XGBoost* 

## Key Implementation Details

- This Getting Started sample code is implemented for CPU using the Python language. The example assumes you have XGboost installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the [Intel® AI Tools](https://software.intel.com/en-us/oneapi/ai-kit).

- XGBoost* is ready for use once you finish the Intel® AI Tools installation and have run the post installation script.

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
>**Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/IntelPython_XGBoost_GettingStarted#environment-setup) is completed.
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
conda activate xgboost
``` 
3. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/IntelPython_XGBoost_GettingStarted
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
IntelPython_XGBoost_GettingStarted.ipynb
```

7. Change the kernel to xgboost
  
8. Run every cell in the Notebook in sequence.

### Conda/PIP
> **Note**: Make sure your Conda/Python environment with AI Tools installed is activated
1. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneapi-samples/AI-and-Analytics/Getting-Started-Samples/IntelPython_XGBoost_GettingStarted
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
IntelPython_XGBoost_GettingStarted.ipynb
```
6. Run every cell in the Notebook in sequence.

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.



## Example Output

>**Note**: Your numbers might be different. 

```
RMSE: 11.113036205909719
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```
## Related Samples

* [Intel® Python XGBoost Daal4py Prediction](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPython_XGBoost_daal4pyPrediction)
* [Intel® Python Scikit-learn Extension Getting Started](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_SKLearn_GettingStarted)


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
