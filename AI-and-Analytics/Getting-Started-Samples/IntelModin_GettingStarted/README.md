# Intel Modin Getting Started
This Getting Started sample code show how to use distributed Pandas using the Modin package. It demonstrates how to use software products that can be found in the [Intel AI Analytics Toolkit powered by oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html). 

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher
| Hardware                          | Intel Atom® Processors; Intel® Core™ Processor Family; Intel® Xeon® Processor Family; Intel® Xeon® Scalable Performance Processor Family
| Software                          | Modin, Intel® AI Analytics Toolkit
| What you will learn               | basic Intel Distribution of Modin programming model for Intel CPU
| Time to complete                  | 5-8 minutes

## Purpose
Modin uses Ray or Dask to provide an effortless way to speed up your Pandas notebooks, scripts, and libraries. Unlike other distributed DataFrame libraries, Modin provides seamless integration and compatibility with existing Pandas code. 

In this sample you will run Modin-accelerated Pandas functions and note the performance gain when compared to "stock" (aka standard) Pandas functions.

## Key Implementation Details
This Getting Started sample code is implemented for CPU using the Python language. The example assumes you have Pandas and Modin installed inside a conda environment, similar to what is directed by the [oneAPI AI Analytics Toolkit powered by oneAPI](https://software.intel.com/content/www/us/en/develop/articles/installing-ai-kit-with-conda.html). 

## License

This code sample is licensed under MIT license

## Building Modin for CPU

Modin is ready for use once you finish the Intel Distribution of Modin installation, and have run the post installation script.

For this sample, you will also have to install the matplotlib module. 

Please install matplotlib with command: 

```
conda install -c intel matplotlib
```

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation, and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.


### Activate conda environment With Root Access

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the `setvars.sh` script and Intel Distribution of Modin environment installation (https://software.intel.com/content/www/us/en/develop/articles/installing-ai-kit-with-conda.html). Then navigate in Linux shell to your oneapi installation path, typically `/opt/intel/oneapi/` when installed as root or sudo, and `~/intel/oneapi/` when not installed as a super user. If you customized the installation folder, the `setvars.sh` file is in your custom folder.

Activate the conda environment with the following command:

#### Linux
```
source activate intel-aikit-modin
```

### Activate conda environment Without Root Access (Optional)

By default, the Intel AI Analytics toolkit is installed in the `oneapi` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### Linux
```
conda create --name user-intel-aikit-modin --clone intel-aikit-modin
```

Then activate your conda environment with the following command:

```
source activate user-intel-aikit-modin
```


### Install Jupyter Notebook

Launch Jupyter Notebook in the directory housing the code example

```
conda install jupyter nb_conda_kernels
```

#### View in Jupyter Notebook


Launch Jupyter Notebook in the directory housing the code example

```
jupyter notebook
```

## Running the Sample

### Run as Jupyter Notebook

Open .ipynb file and run cells in Jupyter Notebook using the "Run" button (see image using "daal4py Hello World" sample)

![Click the Run Button in the Jupyter Notebook](Jupyter_Run.jpg "Run Button on Jupyter Notebook")

### Run as Python File

Open notebook in Jupyter and download as python file (see image using "daal4py Hello World" sample)

![Download as python file in the Jupyter Notebook](Jupyter_Save_Py.jpg "Download as python file in the Jupyter Notebook")

Run the Program

`python IntelModin_GettingStarted.py`

##### Expected Printed Output:
Expected Cell Output shown in IntelModin_GettingStarted.ipynb
