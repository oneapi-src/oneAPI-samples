# Intel® Python daal4py Get Started Sample

This get started sample code shows how to do batch linear regression using the Python API package daal4py powered by the oneAPI Data Analytics Library (oneDAL). It demonstrates how to use software products that are powered by [oneAPI Data Analytics Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) and found in the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Property                          | Description
| :---                              | :---
| Category                          | Get started sample
| What you will learn               | Basic daal4py programming model for Intel CPUs
| Time to complete                  | 5 minutes

## Purpose

daal4py is a simplified API to Intel® oneDAL that allows for fast usage of the framework suited for data scientists or machine learning users. Built to help provide an abstraction to Intel® oneDAL for direct usage or integration into one's own framework.

In this sample, you will run a batch Linear Regression model with oneDAL daal4py library memory objects. You will also learn how to train a model and save the information to a file.

| Optimized for                     | Description
| :---                              | :---
| OS                                | <ul><li>64-bit Linux\*: Ubuntu\* 18.04 or higher</li><li>64-bit Windows\* 10</li><li>macOS* 10.14 or higher</li></ul>
| Hardware                          | <ul><li>Intel Atom® processors</li><li>Intel® Core™ processor family</li><li>Intel® Xeon® processor family</li><li>Intel® Xeon® Scalable processor family</li></ul>
| Software                          | Intel® oneAPI AI Analytics Toolkit

## Key Implementation Details

This get started sample code is implemented for CPUs using the Python language. The example assumes you have daal4py and scikit-learn installed inside a conda environment, similar to what is delivered with the installation of the Intel&reg; Distribution for Python* as part of the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/en-us/oneapi/ai-kit).

## Environment Setup

1. Install Intel® oneAPI AI Analytics Toolkit.

   If you use the Intel&reg; DevCloud, skip this step. The toolkit is
   already installed for you.

   The oneAPI Data Analytics Library is ready for use once you finish the
   Intel® oneAPI AI Analytics Toolkit installation and have run the post
   installation script.

   You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

2. Set up your Intel&reg; oneAPI AI Analytics Toolkit environment.

   Source the `setvars` script located in the root of your oneAPI installation.

   - Linux Sudo: ``. /opt/intel/oneapi/setvars.sh``

   - Linux User: ``. ~/intel/oneapi/setvars.sh``

   - Windows: ``C:\Program Files(x86)\Intel\oneAPI\setvars.bat``

   For more information on environment variables, see [Use the setvars Script for Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


3. Activate the conda environment.

   - If you have the root access to your oneAPI installation path or if you use the Intel&reg; DevCloud:
   
     Intel Python environment will be active by default. However, if you activated another environment, you can return with the following command:

     ``` bash
     source activate base
     ```
	 
   - If you do not have the root access to your oneAPI installation path:

     By default, the Intel® oneAPI AI Analytics Toolkit is installed in the ``/opt/intel/oneapi`` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

     ``` bash
     conda create --name usr_intelpython --clone base
     ```

     Then activate your conda environment with the following command:

     ``` bash
     source activate usr_intelpython
     ```

4. Install Jupyter Notebook.

   If you use the Intel DevCloud, skip this step.

   ``` bash
   conda install jupyter nb_conda_kernels
   ```

## Run the Sample<a name="running-the-sample"></a>

You can run the sample code in a Jupyter notebook or as a Python script locally or in the Intel DevCloud.

### Run the Sample in Jupyter Notebook<a name="run-as-jupyter-notebook"></a>

To open the Jupyter notebook on your local server:

1. Activate the conda environment.

   ``` bash
   source activate base
   # or
   source activate usr_intelpython
   ```

2. Start the Jupyter notebook server.

   ``` bash
   jupyter notebook
   ```
   
3. Open the ``IntelPython_daal4py_GettingStarted.ipynb`` file in the Notebook
   Dashboard.

4. Run the cells in the Jupyter notebook sequentially by clicking the
   **Run** button.

   ![Click the Run button in Jupyter Notebook](Jupyter_Run.jpg "Run button in Jupyter Notebook")

### Run the Python Script Locally

1. Activate the conda environment.

   ``` bash
   source activate base
   # or
   source activate usr_intelpython
   ```

2. Run the Python script.

   ``` bash
   python IntelPython_daal4py_GettingStarted.py
   ```

The script saves the output files in the included ``models`` and ``results`` directories.

#### Expected Printed Output

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


### Run the Sample in the Intel&reg; DevCloud for oneAPI JupyterLab<a name="run-samples-on-devcloud"></a>

1. Open the following link in your browser: https://jupyter.oneapi.devcloud.intel.com/

2. In the Notebook Dashboard, navigate to the ``IntelPython_daal4py_GettingStarted.ipynb`` file and open it.

3. Run the sample code and read the explanations in the notebook.


### Run the Sample in the Intel&reg; DevCloud in Batch Mode

This sample includes the ``run.sh`` script for batch processing.

Submit a job that requests a compute node to run the sample code:

``` bash
qsub -l nodes=1:xeon:ppn=2 -d . run.sh
```
   
<details>
<summary>Click here for additional information about requesting a compute node in the Intel DevCloud.</summary>
   
In order to run a script in the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
   
This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
| __CPU__           | __qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh__      |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |
</details>

The script saves the output files in the included ``models`` and ``results`` directories.

### Run the Sample in Visual Studio Code*

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:

1. Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.

2. Configure the oneAPI environment with the extension **Environment Configurator for Intel(R) oneAPI Toolkits**.

3. Open a Terminal in VS Code by clicking **Terminal** > **New Terminal**.

4. Run the sample in the VS Code terminal using the instructions in this document.

On Linux, you can debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this document for instructions on how to build and run a sample. 

### Related Samples

Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.

### Troubleshooting

If an error occurs, troubleshoot the problem using the [Diagnostics Utility for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
