# `Intel Modin Getting Started` Sample
This Getting Started sample code shows how to use distributed Pandas using the Intel® Distribution of Modin* package. It demonstrates how to use software products that can be found in the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher
| Hardware                          | Intel Atom® Processors; Intel® Core™ Processor Family; Intel® Xeon® Processor Family; Intel® Xeon® Scalable Performance Processor Family
| Software                          | Intel Distribution of Modin*, Intel® oneAPI AI Analytics Toolkit
| What you will learn               | Basic Intel Distribution of Modin* programming model for Intel CPU
| Time to complete                  | 5-8 minutes

## Purpose
Intel Distribution of Modin* uses Ray or Dask to provide an effortless way to speed up your Pandas notebooks, scripts, and libraries. Unlike other distributed DataFrame libraries, Intel Distribution of Modin* provides seamless integration and compatibility with existing Pandas code.

In this sample, you will run Intel Distribution of Modin*-accelerated Pandas functions and note the performance gain when compared to "stock" (aka standard) Pandas functions.

## Key Implementation Details

This Getting Started sample code is implemented for CPU using the Python language. The example assumes you have Pandas and Modin installed inside a conda environment, similar to what is directed by the [Intel® oneAPI AI Analytics Toolkit](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/conda/install-intel-ai-analytics-toolkit-via-conda.html).


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).


## Running Samples on the Intel&reg; DevCloud
If you are running this sample on the DevCloud, see [Running Samples on the Intel&reg; DevCloud](#run-samples-on-devcloud)

## Building Modin for CPU


Intel Distribution of Modin* is ready for use once you finish the Intel Distribution of Modin installation and have run the post installation script.

For this sample, you will also have to install the matplotlib module.

Please install matplotlib with the command:

```
conda install -c intel matplotlib
```

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.


### Activate conda environment With Root Access

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the `setvars.sh` script and [Intel Distribution of Modin environment installation](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/conda/install-intel-ai-analytics-toolkit-via-conda.html). Then navigate in Linux shell to your oneapi installation path, typically `/opt/intel/oneapi/` when installed as root or sudo, and `~/intel/oneapi/` when not installed as a superuser. If you customized the installation folder, the `setvars.sh` file is in your custom folder.

Activate the conda environment with the following command:

#### Linux
```
source activate intel-aikit-modin
```

### Activate conda environment Without Root Access (Optional)

By default, the Intel® oneAPI AI Analytics toolkit is installed in the `oneapi` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### Linux
```
conda create --name user-intel-aikit-modin --clone intel-aikit-modin
```

Then activate your conda environment with the following command:

```
source activate user-intel-aikit-modin
```


### Install Jupyter Notebook

Launch Jupyter Notebook in the directory housing the code example:

```
conda install jupyter nb_conda_kernels
```

#### View in Jupyter Notebook


Launch Jupyter Notebook in the directory housing the code example:

```
jupyter notebook
```

## Running the Sample<a name="running-the-sample"></a>

### Run as Jupyter Notebook<a name="run-as-jupyter-notebook"></a>

Open .ipynb file and run cells in Jupyter Notebook using the "Run" button (see the image using "daal4py Hello World" sample):

![Click the Run Button in the Jupyter Notebook](Jupyter_Run.jpg "Run Button on Jupyter Notebook")

#### Intel® DevCloud for oneAPI JupyterLab

Please note that as of right now, this sample cannot be run on Intel® DevCloud for oneAPI JupyterLab due to conflicts between the Intel® DevCloud for oneAPI JupyterLab platform and Modin dependencies. This is a known issue that Intel is currently working on resolving. Thank you for your patience.

### Run as Python File

Open notebook in Jupyter and download as python file (see the image using "daal4py Hello World" sample):

![Download as python file in the Jupyter Notebook](Jupyter_Save_Py.jpg "Download as python file in the Jupyter Notebook")

Run the Program

`python IntelModin_GettingStarted.py`

##### Expected Printed Output:
Expected Cell Output is shown in IntelModin_GettingStarted.ipynb

### Running Samples on the Intel&reg; DevCloud (Optional)<a name="run-samples-on-devcloud"></a>

<!---Include the next paragraph ONLY if the sample runs in batch mode-->
### Run in Batch Mode
This sample runs in batch mode, so you must have a script for batch processing. Once you have a script set up, refer to [Running the Sample](#running-the-sample).

<!---Include the next paragraph ONLY if the sample DOES NOT RUN in batch mode-->
### Run in Interactive Mode
This sample runs in interactive mode. For more information, see [Run as Juypter Notebook](#run-as-jupyter-notebook).

### Request a Compute Node
In order to run on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
 This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

<!---Mark each compatible Node in BOLD-->
| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
| CPU               | qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh          |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |

