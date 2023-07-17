#  `Census` Sample: End-to-End Machine Learning Workload

The `Census` sample code illustrates how to use Intel® Distribution of Modin* for ETL operations and ridge regression algorithm from the Intel® Extension for Scikit-learn* library to build and run an end-to-end machine learning (ML) workload.

| Area                     | Description
|:---                      | :---
| What you will learn      | How to use Intel Distribution of Modin* and Intel® Extension for Scikit-learn* to build end-to-end ML workloads and gain performance.
| Time to complete         | 20 minutes


## Purpose
This sample code demonstrates how to run the end-to-end census workload using the AI Toolkit without any external dependencies.

Intel® Distribution of Modin* uses HDK to speed up your Pandas notebooks, scripts, and libraries. Unlike other distributed DataFrame libraries, Intel® Distribution of Modin* provides integration and compatibility with existing Pandas code. Intel® Extension for Scikit-learn* dynamically patches scikit-learn estimators to use Intel® oneAPI Data Analytics Library (oneDAL) as the underlying solver to get the solution faster.

## Prerequisites

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Ubuntu* 18.04 or higher
| Hardware                          | Intel Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable processor family
| Software                          | Intel® AI Analytics Toolkit (AI Kit) (Python version 3.8 or newer, Intel® Distribution of Modin*) <br> Intel® Extension for Scikit-learn* <br> NumPy

The Intel® Distribution of Modin* and Intel® Extension for Scikit-learn* libraries are available together in [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).


## Key Implementation Details

This end-to-end workload sample code is implemented for CPU using the Python language. Once you have installed AI Kit, the Conda environment is prepared with Python version 3.8 (or newer), Intel Distribution of Modin*, Intel® Extension for Scikit-Learn, and NumPy. 

In this sample, you will use Intel® Distribution of Modin* to ingest and process U.S. census data from 1970 to 2010 in order to build a ridge regression-based model to find the relation between education and total income earned in the US.

The data transformation stage normalizes the income to yearly inflation, balances the data so each year has a similar number of data points, and extracts the features from the transformed dataset. The feature vectors are input into the ridge regression model to predict the education of each sample.

>**Note**: The dataset is from IPUMS USA, University of Minnesota, [www.ipums.org](https://ipums.org/). <br> Steven Ruggles, Sarah Flood, Ronald Goeken, Josiah Grover, Erin Meyer, Jose Pacas, and Matthew Sobek. IPUMS USA: Version 10.0 \[dataset\]. Minneapolis, MN: IPUMS, 2020. [https://doi.org/10.18128/D010.V10.0](https://doi.org/10.18128/D010.V10.0).


## Configure the Development Environment
If you do not already have the AI Kit installed, then download an online or offline installer for the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) or install the AI Kit using Conda.

>**Note**: See [Install Intel® AI Analytics Toolkit via Conda*](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/conda/install-intel-ai-analytics-toolkit-via-conda.html) in the *Intel® oneAPI Toolkits Installation Guide for Linux* OS* for information on Conda installation and configuration.

The Intel® Distribution of Modin* and the Intel® Extension for Scikit-learn* are ready to use after AI Kit installation with the Conda Package Manager.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.
 5. (Linux only) Debug GPU applications with GDB for Intel® oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions and how to configure the oneAPI environment, see the [Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## Configure the Environment

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

1. Install the Intel® Distribution of Modin* python environment (Only python 3.8 - 3.10 are supported).
   ```
   conda create -n modin-hdk python=3.x -y
   ```
2. Activate the Conda environment.
   ```
   conda activate modin-hdk
   ```
3. Install modin-hdk, Intel® Extension for Scikit-learn* and related libraries.
   ```
   conda install modin-hdk -c conda-forge -y
   pip install scikit-learn-intelex
   pip install matplotlib
   ```
4. Install Jupyter Notebook
   ```
   pip install jupyter ipykernel
   ```
5. Add kernel to Jupyter Notebook.
   ```
   python -m ipykernel install --user --name modin-hdk
   ```
6. Change to the sample directory, and open Jupyter Notebook.
   ```
   jupyter notebook
   ```

## Run the `Census` Sample

### Run the Jupyter Notebook

1. Open `census_modin.ipynb`.
2. Click **Run** to run the cells.

   ![Click the Run Button in the Jupyter Notebook](Running_Jupyter_notebook.jpg "Run Button on Jupyter Notebook")

3. Alternatively, run the entire workbook by selecting **Restart kernel and re-run whole notebook**.

### Run as Python File
1. Open the notebook in Jupyter.
2. From the **File** menu, select **Download as** > **Python (.py)**.

   ![Download as python file in the Jupyter Notebook](Running_Jupyter_notebook_as_Python.jpg "Download as python file in the Jupyter Notebook")

3. Run the program.
   ```
   python census_modin.py
   ```

   #### Troubleshooting
   If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


### Run Notebook in Intel® DevCloud in JupyterLab*

1. If you do not already have an account, request an Intel® DevCloud account at [Create an Intel® DevCloud Account](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).
2. Open a web browser, and navigate to https://devcloud.intel.com. Select **Work with oneAPI**.
3. From Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started), locate the ***Connect with Jupyter* Lab*** section (near the bottom).
4. Click **Sign in to Connect** button. (If you are already signed in, the link should say ***Launch JupyterLab****.)
5. Open a terminal from Launcher
6. Follow [step 1-5](#on-linux) to create conda environment
7. Clone the samples from GitHub. If the samples are already present, skip this step.
   ```
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```
8. Change to the sample directory.
9. Open `census_modin.ipynb`.
10. Select kernel "modin-hdk" 
11. Click **Run** to run the cells.
12. Alternatively, run the entire workbook by selecting **Restart kernel and re-run whole notebook**.

## Example Output

This is an example Cell Output for `census_modin.ipynb` run in Jupyter Notebook. 
![Output](Expected_output.jpg "Expected output for Jupyter Notebook")


## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
