# `Intel® Python Daal4py Getting Started` Sample

The `Intel® Python Daal4py Getting Started` sample code shows how to do batch linear regression using the Python API package daal4py powered by the Intel® oneAPI Data Analytics Library (oneDAL).

| Area                   | Description
| :---                   | :---
| What you will learn    | Basic daal4py programming model for Intel CPUs
| Time to complete       | 5 minutes
| Category               | Getting Started

The sample demonstrates how to use software products that are powered by the [Intel® oneAPI Data Analytics Library (oneDAL)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) and some components found in the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

## Purpose

daal4py is a simplified API to oneDAL that allows for fast usage of the framework suited for data scientists or machine learning users. Built to help provide an abstraction to Intel® oneDAL for direct usage or integration into one's own framework.

In this sample, you will run a batch Linear Regression model with oneDAL daal4py library memory objects. You will also learn how to train a model and save the information to a file.

## Prerequisites

| Optimized for           | Description
| :---                    | :---
| OS                      | Ubuntu* 20.04 (or newer)
| Hardware                | Intel Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable processor family
| Software                | Intel® AI Analytics Toolkit (AI Kit) <br> Intel® oneAPI Data Analytics Library (oneDAL)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/) for information.

## Key Implementation Details

This get started sample code is implemented for CPUs using the Python language. The example assumes you have daal4py and scikit-learn installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the Intel® AI Analytics Toolkit.

The Intel® oneAPI Data Analytics Library (oneDAL) is ready for use once you finish the Intel® AI Analytics Toolkit installation and have run the post installation script.

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

### On Linux*

1. Activate the conda environment.

   1. If you have the root access to your oneAPI installation path, choose this option.
   
      Intel Python environment will be active by default. However, if you activated another environment, you can return with the following command.
      ```
      source activate base
      ```
	 
   2. If you do not have the root access to your oneAPI installation path, choose this option.

      By default, the Intel® AI Analytics Toolkit is installed in the ``/opt/intel/oneapi`` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment and activate it using the following commands.
      ```
      conda create --name usr_intelpython --clone base
      source activate usr_intelpython
      ```

2. Install Jupyter Notebook. (Skip this step for Intel® DevCloud.)
   ```
   conda install jupyter nb_conda_kernels
   ```

## Run the `Intel® Python Daal4py Getting Started` Sample

You can run the sample code in a Jupyter Notebook or as a Python script locally or in Intel® DevCloud.

### In Jupyter Notebook

1. Activate the conda environment.
   ```
   source activate base
   # or
   source activate usr_intelpython
   ```

2. Start the Jupyter notebook server.
   ```
   jupyter notebook
   ```

3. Locate and select the Notebook.
   ```
   IntelPython_daal4py_GettingStarted.ipynb
   ```
4. Click the **Run** button to execute all cells in the Notebook in sequence.

### Run the Python Script Locally

1. Activate the conda environment.
   ```
   source activate base
   # or
   source activate usr_intelpython
   ```

2. Run the Python script.
   ```
   python IntelPython_daal4py_GettingStarted.py
   ```

   The script saves the output files in the included ``models`` and ``results`` directories.

### On Intel® DevCloud

#### Using oneAPI JupyterLab

1. Navigate to the JupyterHub for Intel® DevCloud: [jupyter.oneapi.devcloud.intel.com](https://jupyter.oneapi.devcloud.intel.com/).

2. In the Dashboard, locate and navigate to the `IntelPython_daal4py_GettingStarted.ipynb`.

3. Open the Jupyter Notebook.

4. Click the **Run** button to execute all cells in the Notebook in sequence. (Read the explanations of the code in the cells.)

#### In Batch Mode

This sample includes the ``run.sh`` script for batch processing.

Submit a job that requests a compute node to run the sample code:

```
qsub -l nodes=1:xeon:ppn=2 -d . run.sh
```

<details>
<summary>You can specify nodes using a single line script.</summary>

```
qsub  -I  -l nodes=1:xeon:ppn=2 -d . run.sh
```
- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:xeon:ppn=2` (lower case L) assigns one full GPU node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes    |Command Options
  |:---               |:---
  |GPU	             |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	             |`qsub -l nodes=1:xeon:ppn=2 -d .`

  >**Note**: For more information on how to specify compute nodes read *[Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/)* in  the Intel® DevCloud Documentation.
</details>

The script saves the output files in the included `models` and `results` directories.

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

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).