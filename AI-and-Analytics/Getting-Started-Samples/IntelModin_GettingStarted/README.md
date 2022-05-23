# Intel&reg; Modin* Get Started Sample

This get started sample code shows how to use distributed Pandas using the Intel® Distribution of Modin* package. It demonstrates how to use software products that can be found in the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Property                          | Description
| :---                              | :---
| Category                          | Get started sample
| What you will learn               | Basic Intel&reg; Distribution of Modin* programming model for Intel processors
| Time to complete                  | 5-8 minutes


## Purpose

Intel Distribution of Modin* uses Ray or Dask to provide an effortless way to speed up your Pandas notebooks, scripts, and libraries. Unlike other distributed DataFrame libraries, Intel Distribution of Modin* provides seamless integration and compatibility with existing Pandas code.

In this sample, you will run Intel Distribution of Modin*-accelerated Pandas functions and note the performance gain when compared to "stock" (aka standard) Pandas functions.

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher
| Hardware                          | Intel® Atom® processors; Intel® Core™ processor family; Intel® Xeon® processor family; Intel® Xeon® Scalable Performance processor family
| Software                          | Intel® Distribution of Modin*, Intel® oneAPI AI Analytics Toolkit


## Key Implementation Details

This get started sample code is implemented for CPU using the Python language. The example assumes you have Pandas and Modin installed inside a conda environment.


## Environment Setup

1. Install Intel Distribution of Modin in a new conda environment.

   <!-- As of right now, you can install Intel Distribution of Modin only via Anaconda. -->

   ``` bash
   conda create -n aikit-modin
   conda activate aikit-modin
   conda install modin-all -c intel -y
   ```
   
   <!-- You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts. -->

   
2. Install matplotlib.

   ``` bash
   conda install -c intel matplotlib -y
   ```
   
3. Install Jupyter Notebook.

   Skip this step if you are working on the Intel DevCloud.

   ``` bash
   conda install jupyter nb_conda_kernels -y
   ```

4. Create a new kernel for Jupyter Notebook based on your activated conda environment.

   ``` bash
   conda install ipykernel
   python -m ipykernel install --user --name usr_modin
   ```
   
   This step is optional if you plan to open the notebook on your local server.


## Run the Sample<a name="running-the-sample"></a>

You can run the Jupyter notebook with the sample code on your local
server or download the sample code from the notebook as a Python file and run it locally or on the Intel DevCloud.

**Note:** You can run this sample on the Intel DevCloud using the Dask and OmniSci engine backends for Modin. To learn how to set the engine backend for Intel Distribution of Modin, visit the [Intel® Distribution of Modin Getting Started Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-distribution-of-modin-getting-started-guide.html). The Ray backend cannot be used on Intel DevCloud at this time. Thank you for your patience.

### Run the Sample in Jupyter Notebook<a name="run-as-jupyter-notebook"></a>

To open the Jupyter notebook on your local server:

1. Activate the conda environment.

   ``` bash
   conda activate aikit-modin
   ```

2. Start the Jupyter notebook server.

   ``` bash
   jupyter notebook
   ```
   
3. Open the ``IntelModin_GettingStarted.ipynb`` file in the Notebook
   Dashboard.

4. Run the cells in the Jupyter notebook sequentially by clicking the
   **Run** button.

   ![Click the Run button in Jupyter Notebook](Jupyter_Run.jpg "Run button in Jupyter Notebook")

### Run the Sample in the Intel® DevCloud for oneAPI JupyterLab

1. Open the following link in your browser: https://jupyter.oneapi.devcloud.intel.com/

2. In the Notebook Dashboard, navigate to the ``IntelModin_GettingStarted.ipynb`` file and open it.

   **Important:** You must edit the cell that imports modin to enable the Dask or OmniSci backend engine. The Ray backend cannot be used on Intel DevCloud at this time. For more information, visit the [Intel® Distribution of Modin Getting Started Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-distribution-of-modin-getting-started-guide.html).

3. To change the kernel, click **Kernel** > **Change kernel** > **usr_modin**.

4. Run the sample code and read the explanations in the notebook.

### Run the Python Script Locally

1. Convert ``IntelModin_GettingStarted.ipynb`` to a python file in one of the following ways:

   - Open the notebook in Jupyter and download as a python file. See the image from the daal4py Hello World sample:

     ![Download as a python script in Jupyter Notebook](Jupyter_Save_Py.jpg "Download as Python script in the Jupyter Notebook")
	 
   - Run the following command to convert the notebook file to a Python script:
   
     ``` bash
     jupyter nbconvert --to python IntelModin_GettingStarted.ipynb
     ```

2. Run the Python script.

   ``` bash
   ipython IntelModin_GettingStarted.py
   ```

### Run the Sample on the Intel&reg; DevCloud in Batch Mode<a name="run-samples-on-devcloud"></a>

This sample runs in batch mode, so you must have a script for batch processing.

1. Convert ``IntelModin_GettingStarted.ipynb`` to a python file.

   ``` bash
   jupyter nbconvert --to python IntelModin_GettingStarted.ipynb
   ```

2. Create a shell script file ``run-modin-sample.sh`` to activate the conda environment and run the sample.

   ```bash
   source activate aikit-modin
   ipython IntelModin_GettingStarted.py
   ```
   
3. Submit a job that requests a compute node to run the sample code.

   ```bash
   qsub -l nodes=1:xeon:ppn=2 -d . run-modin-sample.sh -o output.txt
   ```
   
   The ``-o output.txt`` option redirects the output of the script to the ``output.txt`` file.

   <details>
   <summary>Click here for additional information about requesting a compute node in the Intel DevCloud.</summary>
   
   In order to run a script on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
   
   This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

   <!---Mark each compatible Node in BOLD-->
   | Node              | Command                                                 |
   |-------------------|---------------------------------------------------------|
   | GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
   | CPU               | qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh          |
   | FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
   | FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |
   </details>

### Run the Sample in Visual Studio Code*

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:

1. Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.

2. Configure the oneAPI environment with the extension **Environment Configurator for Intel(R) oneAPI Toolkits**.

3. Open a Terminal in VS Code by clicking **Terminal** > **New Terminal**.

4. Run the sample in the VS Code terminal using the instructions below.

On Linux, you can debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### Expected Printed Output:

Expected cell output is shown in IntelModin_GettingStarted.ipynb.

## Related Samples

Several sample programs are available for you to try, many of which
can be compiled and run in a similar fashion. Experiment with running
the various samples on different kinds of compute nodes or adjust
their source code to experiment with different workloads.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
