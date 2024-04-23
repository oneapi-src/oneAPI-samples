# `Intel® Python XGBoost Daal4py Prediction` Sample

This sample code illustrates how to analyze the performance benefit of minimal code changes to port pre-trained XGBoost model to daal4py prediction for much faster prediction. 

| Area                   | Description
| :---                   | :---
| What you will learn    | How to analyze the performance benefit of minimal code changes to port pre-trained XGBoost model to daal4py prediction for much faster prediction
| Time to complete       | 5-8 minutes
| Category               | Code Optimization

## Purpose

This sample illustrates how to analyze the performance benefit of minimal code changes to port pre-trained XGBoost models to Daal4py prediction for much faster prediction.

XGBoost is a widely used gradient boosting library in the classical machine learning (ML) area. Designed for flexibility, performance, and portability, XGBoost includes optimized distributed gradient boosting frameworks and implements Machine Learning algorithms underneath. In addition, Daal4py provides functionality to bring even more optimizations to gradient boosting prediction with XGBoost without modifying XGBoost models or learning an additional API.

## Prerequisites

| Optimized for   | Description
|:---             |:---
| OS              | Ubuntu* 18.04 or higher
| Hardware        | Intel Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable processor family
| Software        | XGBoost model <br> Intel® AI Analytics Toolkit (AI Kit)

This sample code is implemented for CPU using the Python language. The sample assumes you have XGboost, Daal4py, and Matplotlib installed inside a conda environment. Installing the Intel® Distribution for Python* as part of the [Intel® AI Analytics Toolkit](https://software.intel.com/en-us/oneapi/ai-kit) should suffice in most cases.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

## Key Implementation Details

In this sample, you will run an XGBoost model with Daal4py prediction and XGBoost API prediction to see the performance benefit of Daal4py gradient boosting prediction. You will also learn how to port a pre-trained XGBoost model to Daal4py prediction.

XGBoost* is ready for use once you finish the Intel® AI Analytics Toolkit installation and have run the post installation script.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Intel® Python XGBoost Daal4py Prediction` Sample

### On Linux*

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

#### Activate Conda with Root Access

By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it. However, if you activated another environment, you can return with the following command.

```
source activate base
```

#### Activate Conda without Root Access (Optional)

You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

```
conda create --name usr_intelpython --clone base
source activate usr_intelpython
```

#### Install Jupyter Notebook

```
conda install jupyter nb_conda_kernels
```

#### Open Jupyter Notebook

> **Note**: This distributed sample cannot be executed from the Jupyter Notebook, but you can read the description and follow the program flow in the Notebook.

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook
   ```
3. Locate and select the Notebook.
   ```
   IntelPython_XGBoost_daal4pyPrediction.ipynb
   ```
4. Click the **Run** button to move through the cells in sequence.

#### Download and Run the Script

1. Still in Jupyter Notebook.

2. Select **Download as** > **python (py)**.

3. Locate the downloaded script.

4. Run the script.
   ```
   python IntelPython_XGBoost_daal4pyPrediction.py
   ```
   When it finishes, you will see two plots: one for prediction time and one for prediction accuracy. You might need to dismiss the first plot to view the second plot.

## Example Output

In addition to the plots for prediction time and prediction accuracy, you should see output similar to the following example.

```
XGBoost prediction results (first 10 rows):
 [4. 2. 2. 2. 3. 1. 3. 4. 3. 4.]

daal4py prediction results (first 10 rows):
 [4. 2. 2. 2. 3. 1. 3. 4. 3. 4.]

Ground truth (first 10 rows):
 [4. 2. 2. 2. 3. 1. 3. 4. 3. 4.]
XGBoost errors count: 10
XGBoost accuracy score: 0.99

daal4py errors count: 10
daal4py accuracy score: 0.99

 XGBoost Prediction Time: 0.03896141052246094

 daal4py Prediction Time: 0.10008668899536133

All looks good!
speedup: 0.3892766452116991
Accuracy Difference 0.0
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).