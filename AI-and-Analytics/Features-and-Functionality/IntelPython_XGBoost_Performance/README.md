# `Intel® Python XGBoost Performance` Sample

This `Intel® Python XGBoost Performance` sample illustrates how to analyze the performance benefit from using Intel optimizations upstreamed by Intel to latest XGBoost compared to un-optimized XGBoost 0.81. It demonstrates how to use software products that can be found in the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Area                 | Description
| :---                 | :---
| What you will learn  | How to analyze the performance benefit from using Intel optimizations upstreamed by Intel to the latest XGBoost compared to un-optimized XGBoost 0.81
| Time to complete     | 10-15 minutes
| Category             | Code Optimization

## Purpose

XGBoost is a widely used gradient boosting library in the classical machine learning (ML) area. Designed for flexibility, performance, and portability, XGBoost includes optimized distributed gradient boosting frameworks and implements Machine Learning algorithms underneath. 

## Prerequisites

| Optimized for   | Description
|:---             |:---
| OS              | Ubuntu* 18.04 or higher
| Hardware        | Intel Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable processor family
| Software        | XGBoost <br> Intel® AI Analytics Toolkit (AI Kit)

## Key Implementation Details

In this sample, you will use a XGBoost model and prediction using Intel optimizations upstreamed by Intel to the latest XGBoost package and the un-optimized XGBoost 0.81 for comparison.

This XGBoost sample code is implemented for the CPU using the Python language. The example assumes you XGBoost installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the [Intel® AI Analytics Toolkit](https://software.intel.com/en-us/oneapi/ai-kit). It also assumes you have set up an additional XGBoost 0.81 conda environment, with details on how to do so explained within the sample and this README.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Intel® Python XGBoost Performance` Sample

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

XGBoost is ready for use once you finish the Intel® AI Analytics Toolkit installation and have run the post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

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

### Build Another XGBoost 0.81 Conda Environment

To see the performance comparison to the latest XGBoost with Intel optimizations and un-optimized XGBoost 0.81, you must run this sample in a second conda environment with **XGBoost 0.81** installed.

See the information in the Jupyter Notebook on how to set up and run the different conda environments.

### View in Jupyter Notebook

> **Note**: This distributed sample cannot be executed from the Jupyter Notebook, but you can read the description and follow the program flow in the Notebook.

#### Open Jupyter Notebook

> **Note**: This distributed sample cannot be executed from the Jupyter Notebook, but you can read the description and follow the program flow in the Notebook.

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook
   ```
3. Locate and select the Notebook.
   ```
   IntelPython_XGBoost_Performance.ipynb
   ```
4. Click the **Run** button to move through the cells in sequence.

### Run the Python Script

1. Change to the sample directory.
2. Locate and run the script.
   ```
   python IntelPython_XGBoost_Performance.py
   ```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

```
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).