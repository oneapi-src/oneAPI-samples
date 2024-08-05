# Modin* Get Started Sample

The `Modin* Getting Started` sample demonstrates how to use distributed Pandas using the Modin package. 

| Area                  | Description
| :---                  | :---
| Category              | Getting Started
| What you will learn   | Basic Modin* programming model for Intel processors
| Time to complete      | 5 to 8 minutes

## Purpose

Modin uses Ray or Dask to provide a method to speed up your Pandas notebooks, scripts, and libraries. Unlike other distributed DataFrame libraries, Modin provides integration and compatibility with existing Pandas code.

In this sample, you will run Modin-accelerated Pandas functions and note the performance gain when compared to "stock" (or standard) Pandas functions.

## Prerequisites

| Optimized for                     | Description
| :---                              | :---
| OS                                | Ubuntu* 18.04 (or newer)
| Hardware                          | Intel® Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable Performance processor family
| Software                          | Modin 

## Key Implementation Details

This get started sample code is implemented for CPU using the Python language. The example assumes you have Pandas and Modin installed inside a conda environment.

## Environment Setup

1. Install Modin in a new conda environment.

   >**Note:** replace python=3.x with your own Python version
   ```
   conda create -n modin python=3.x -y
   conda activate modin
   conda install modin-all -c conda-forge -y
   ```

2. Install Matplotlib.
   ```
   conda install -c conda-forge matplotlib -y
   ```

3. Install Jupyter Notebook. 
   ```
   conda install jupyter nb_conda_kernels -y
   ```

4. Create a new kernel for Jupyter Notebook based on your activated conda environment. (This step is optional if you plan to open the Notebook on your local server.)
   ```
   conda install ipykernel
   python -m ipykernel install --user --name usr_modin
   ```
## Run the `Modin* Get Started` Sample

You can run the Jupyter notebook with the sample code on your local server or download the sample code from the notebook as a Python file and run it locally. 

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
   Modin_GettingStarted.ipynb
   ```

4. Click the **Run** button to move through the cells in sequence.

### Run the Python Script Locally

1. Convert ``Modin_GettingStarted.ipynb`` to a Python file. There are two options.

   1. Open the notebook and download the script as Python file: **File > Download as > Python (py)**.

   2. Convert the notebook file to a Python script using commands similar to the following.
      ```
      jupyter nbconvert --to python Modin_GettingStarted.ipynb
      ```
2. Run the Python script.
   ```
   ipython Modin_GettingStarted.py
   ```

### Expected Output

The expected cell output is shown in the `Modin_GettingStarted.ipynb` Notebook.

## Related Samples

* [Modin Vs. Pandas Performance](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/Modin_Vs_Pandas)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
