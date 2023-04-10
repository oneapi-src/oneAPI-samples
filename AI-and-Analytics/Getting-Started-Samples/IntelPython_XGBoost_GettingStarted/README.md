# `Intel® Python XGBoost* Getting Started` Sample

The `Intel® Python XGBoost* Getting Started` sample demonstrates how to set up and train an XGBoost model on datasets for prediction.

| Area                     | Description
| :---                     | :---
| What you will learn      | The basics of XGBoost programming model for Intel CPUs
| Time to complete         | 5 minutes
| Category                 | Getting Started

## Purpose

XGBoost* is a widely used gradient boosting library in the classical ML area. Designed for flexibility, performance, and portability, XGBoost* includes optimized distributed gradient boosting frameworks and implements Machine Learning algorithms underneath. Starting with 0.9 version of XGBoost, Intel has been up streaming optimizations through the `hist` histogram tree-building method. Starting with 1.3.3 version of XGBoost and beyond, Intel has also begun up streaming inference optimizations to XGBoost as well.

In this code sample, you will learn how to use Intel optimizations for XGBoost published as part of Intel® AI Analytics Toolkit. The sample also illustrates how to set up and train an XGBoost* model on datasets for prediction. It also demonstrates how to use software products that can be found in the [Intel® AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

## Prerequisites

| Optimized for           | Description
| :---                    | :---
| OS                      | Ubuntu* 20.04 (or newer)
| Hardware                | Intel Atom® Processors <br> Intel® Core™ Processor Family <br> Intel® Xeon® Processor Family <br> Intel® Xeon® Scalable processor family
| Software                | XGBoost* <br> Intel® AI Analytics Toolkit (AI Kit)

## Key Implementation Details

This Getting Started sample code is implemented for CPU using the Python language. The example assumes you have XGboost installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the [Intel® AI Analytics Toolkit](https://software.intel.com/en-us/oneapi/ai-kit).

XGBoost* is ready for use once you finish the Intel® AI Analytics Toolkit installation and have run the post installation script.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Configure Environment

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### Activate Conda with Root Access

If you activated another environment, you can return with the following command:
```
source activate base
```
### Activate Conda without Root Access (Optional)

By default, the Intel® AI Analytics Toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone and active your desired conda environment using the following commands:
```
conda create --name user_base --clone base
source activate user_base
```

## Run the `Intel® Python XGBoost* Getting Started` Sample

### Install Jupyter Notebook

1. Change to the sample directory.
2. Install Jupyter Notebook with an appropriate kernel.
   ```
   conda install jupyter nb_conda_kernels
   ```
### Open Jupyter Notebook

>**Note**: You cannot execute the sample in Jupyter Notebook, but you can still view inside the notebook to follow the included write-up and description.

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook
   ```
3. Locate and select the Notebook.
   ```
   IntelPython_XGBoost_GettingStarted.ipynb
   ```
4. Click the **Run** button to move through the cells in sequence.

### Run the Python Script

1. Still in Jupyter Notebook.

2. Select **File** > **Download as** > **Python (py)**.
3. Run the script.
   ```
   python IntelPython_XGBoost_GettingStarted.py
   ```
   The output files of the script will be saved in **models** and **result** directories.

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Example Output

>**Note**: Your numbers might be different. 

```
RMSE: 11.113036205909719
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).