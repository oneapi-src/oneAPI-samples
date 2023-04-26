# `Intel® Modin* Get Started` Sample

The `Intel® Modin Getting Started` sample demonstrates how to use distributed Pandas using the Intel® Distribution of Modin* package. It demonstrates how to use software products that can be found in the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Area                  | Description
| :---                  | :---
| What you will learn   | Basic Intel® Distribution of Modin* programming model for Intel processors
| Time to complete      | 5 to 8 minutes
| Category              | Getting Started

## Purpose

Intel® Distribution of Modin* uses Ray or Dask to provide a method to speed up your Pandas notebooks, scripts, and libraries. Unlike other distributed DataFrame libraries, Intel® Distribution of Modin* provides integration and compatibility with existing Pandas code.

In this sample, you will run Intel® Distribution of Modin*-accelerated Pandas functions and note the performance gain when compared to "stock" (or standard) Pandas functions.

## Prerequisites

| Optimized for                     | Description
| :---                              | :---
| OS                                | Ubuntu* 18.04 (or newer)
| Hardware                          | Intel® Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable Performance processor family
| Software                          | Intel® Distribution of Modin* <br> Intel® AI Analytics Toolkit (AI Kit)

## Key Implementation Details

This get started sample code is implemented for CPU using the Python language. The example assumes you have Pandas and Modin installed inside a conda environment.

## Configure Environment

1. Install Intel® Distribution of Modin* in a new conda environment.

   >**Note:** replace python=3.x with your own Python version
   ```
   conda create -n aikit-modin python=3.x -y
   conda activate aikit-modin
   conda install modin-all -c intel -y
   ```

2. Install Matplotlib.
   ```
   conda install -c intel matplotlib -y
   ```

3. Install Jupyter Notebook. (Skip this step if you are working on Intel® DevCloud.)
   ```
   conda install jupyter nb_conda_kernels -y
   ```

4. Create a new kernel for Jupyter Notebook based on your activated conda environment. (This step is optional if you plan to open the Notebook on your local server.)
   ```
   conda install ipykernel
   python -m ipykernel install --user --name usr_modin
   ```
## Run the `Intel® Modin* Get Started` Sample

You can run the Jupyter notebook with the sample code on your local server or download the sample code from the notebook as a Python file and run it locally or on the Intel DevCloud. Visit [Intel® Distribution of Modin Getting Started Guide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-distribution-of-modin-getting-started-guide.html) for more information.

### Run the Sample in Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:

1. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
2. Configure the oneAPI environment with the extension **Environment Configurator for Intel(R) oneAPI Toolkits**.
3. Open a Terminal in VS Code by clicking **Terminal** > **New Terminal**.
4. Run the sample in the VS Code terminal using the instructions below.

On Linux, you can debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


### In Jupyter Notebook

1. Activate the conda environment.
   ```
   conda activate aikit-modin
   ```

2. Start the Jupyter Notebook server.
   ```
   jupyter notebook
   ```

3. Locate and open the Notebook.
   ```
   IntelModin_GettingStarted.ipynb
   ```

4. Click the **Run** button to move through the cells in sequence.

### Run the Python Script Locally

1. Convert ``IntelModin_GettingStarted.ipynb`` to a Python file. There are two options.

   1. Open the notebook and download the script as Python file: **File > Download as > Python (py)**.

   2. Convert the notebook file to a Python script using commands similar to the following.
      ```
      jupyter nbconvert --to python IntelModin_GettingStarted.ipynb
      ```
2. Run the Python script.
   ```
   ipython IntelModin_GettingStarted.py
   ```

### Run in Intel® DevCloud (Optional)

#### Using JupyterLab

1. If you do not already have an account, request an Intel® DevCloud account at [Create an Intel® DevCloud Account](https://www.intel.com/content/www/us/en/forms/idz/devcloud-registration.html).

2. Navigate to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started).

3. Locate the **Connect with Jupyter Lab\*** section (near the bottom). 

4. Click **Sign in to Connect** button. (If you are already signed in, the link should say **Launch JupyterLab\***.)
  
5. If the samples are not already present in your Intel® DevCloud account, download them.
   - From JupyterLab, select **File > New > Terminal**.
   - In the terminal, clone the samples from GitHub: 
      ```
      git clone https://github.com/oneapi-src/oneAPI-samples.git
      ```
6. Set up environment in the terminal.
   1. source oneAPI conda environment.
      ```
      source /opt/intel/oneapi/setvars.sh --force
      ```
   2. See [Configure Environment](#configure-environment) to set up the environment properly.
 
1. In the JupyterLab, navigate to the ``IntelModin_GettingStarted.ipynb`` file and open it.

2. Change the kernel. Click **Kernel** > **Change kernel** > **usr_modin**.

3. Run the sample code and read the explanations in the notebook.


#### Using Batch Mode (Optional)

This sample runs in batch mode, so you must have a script for batch processing.

1. Convert ``IntelModin_GettingStarted.ipynb`` to a python file.
   ```
   jupyter nbconvert --to python IntelModin_GettingStarted.ipynb
   ```

2. Create a shell script file ``run-modin-sample.sh`` to activate the conda environment and run the sample.
   ```
   source activate aikit-modin
   ipython IntelModin_GettingStarted.py
   ```

3. Submit a job that requests a compute node to run the sample code.
   ```
   qsub -l nodes=1:xeon:ppn=2 -d . run-modin-sample.sh -o output.txt
   ```
   <details>
   <summary>You can specify other nodes using a single line qsub script.</summary>

   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node.
   - `-d .` makes the current folder as the working directory for the task.

     |Available Nodes    |Command Options
     |:---               |:---
     |GPU	             |`qsub -l nodes=1:gpu:ppn=2 -d .`
     |CPU	             |`qsub -l nodes=1:xeon:ppn=2 -d .`

     >**Note**: For more information on how to specify compute nodes read *[Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/)* in the Intel® DevCloud Documentation.
     </details>
   
   The ``-o output.txt`` option redirects the output of the script to the ``output.txt`` file.

### Expected Output

The expected cell output is shown in the `IntelModin_GettingStarted.ipynb` Notebook.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).