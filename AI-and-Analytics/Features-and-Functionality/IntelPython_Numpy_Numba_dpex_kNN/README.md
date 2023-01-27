# `Intel® Python NumPy vs Numba_dpex` Sample

The `Intel® Python NumPy vs Numba_dpex` sample shows how to achieve the same accuracy of the k-NN model classification while using numpy, numba, and numba_dpex.

| Area                    | Description
| :---                    | :---
| What you will learn     | How to program using numba_dpex
| Time to complete        | 5 minutes
| Category                | Component

>**Note**: The libraries used in this sample are available in Intel® Distribution for Python* as part of the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/en-us/oneapi/ai-kit).

## Purpose

In this sample, you will run a k-nearest neighbors algorithm using 3 different Intel® Distribution for Python* libraries: numpy, numba, and numba_dpex. You will learn how to use k-NN model and how to optimize it by numba_dpex operations without sacrificing accuracy.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04
| Hardware                | CPU
| Software                | Intel® AI Analytics Toolkit

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/) for information.

## Key Implementation Details

This sample code is implemented for the CPU using Python. The sample assumes you have numba_dpex installed inside a Conda environment, similar to what is installed with the Intel® Distribution for Python*.

>**Note**: Read *[Get Started with the Intel® AI Analytics Toolkit for Linux*](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html)* to find out how you can achieve performance gains for popular deep-learning and machine-learning frameworks through Intel optimizations.

## Run the `Intel® Python NumPy vs Numba_dpex` Sample

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

#### Activate Conda

1. Activate the Conda environment.
   ```
   conda activate base
   ```
   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

   ```
   conda create --name usr_base --clone base
   conda activate usr_base
   ```

#### Run the Jupyter Notebook

1. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0
   ```
2. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the Notebook.
   ```
   IntelPython_Numpy_Numba_dpex_kNN.ipynb
   ```
4. Run every cell in the Notebook in sequence.

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

### Run the Sample on Intel® DevCloud

1. If you do not already have an account, request an Intel® DevCloud account at [*Create an Intel® DevCloud Account*](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).
2. On a Linux* system, open a terminal.
3. SSH into Intel® DevCloud.
   ```
   ssh DevCloud
   ```
   > **Note**: You can find information about configuring your Linux system and connecting to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started).

4. Locate and select the Notebook.
   ```
   numba_numpy.ipynb
   ```
5. Run every cell in the Notebook in sequence.

## Example Output

```
Numpy accuracy: 0.7222222222222222

Numba accuracy: 0.7222222222222222

Numba_dpex accuracy 0.7222222222222222

[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).