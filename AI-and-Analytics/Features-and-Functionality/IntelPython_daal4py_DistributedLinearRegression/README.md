# `Intel Python daal4py Distributed Linear Regression Sample`
This sample code shows how to train and predict with a distributed linear regression model using the python API package daal4py for oneAPI Data Analytics Library. It assumes you have a working version of the MPI library installed, and it demonstrates how to use software products that can be found in the [Intel oneAPI Data Analytics Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) or [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html). 

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher, 64-bit Windows 10, macOS 10.14 or higher
| Hardware                          | Intel Atom® Processors; Intel® Core™ Processor Family; Intel® Xeon® Processor Family; Intel® Xeon® Scalable Performance Processor Family
| Software                          | oneDAL Software Library, Python version >= 3.6, conda-build version >= 3, C++ compiler with C++11 support, Pickle, Pandas, NumPy
| What you will learn               | distributed oneDAL Linear Regression programming model for Intel CPU
| Time to complete                  | 5 minutes

## Purpose

daal4py is a simplified API to Intel® oneDAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users. Built to help provide an abstraction to Intel® oneDAL for direct usage or integration into one's own framework.

In this sample, you will run a distributed Linear Regression model with oneDAL daal4py library memory objects. You will also learn how to train a model and save the information to a file.
  
## Key Implementation Details 
This distributed linear regression sample code is implemented for the CPU using the Python language. The example assumes you have daal4py and scikit-learn installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python as part of the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/en-us/oneapi/ai-kit). 
 
## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Running Samples on the Intel&reg; DevCloud
If you are running this sample on the DevCloud, see [Running Samples on the Intel&reg; DevCloud](#run-samples-on-devcloud)

## Building daal4py for CPU

oneAPI Data Analytics Library is ready for use once you finish the Intel® oneAPI AI Analytics Toolkit installation and have run the post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.


### Activate conda environment With Root Access

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the `setvars.sh` script. Then navigate in Linux shell to your oneapi installation path, typically `/opt/intel/oneapi/` when installed as root or sudo, and `~/intel/oneapi/` when not installed as a superuser. If you customized the installation folder, the `setvars.sh` file is in your custom folder. 

Intel Python environment will be active by default. However, if you activated another environment, you can return with the following command:

#### On a Linux* System
```
source activate base
```

### Activate conda environment Without Root Access (Optional)

By default, the Intel® oneAPI AI Analytics Toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### On a Linux* System
```
conda create --name usr_intelpython --clone base
```

Then activate your conda environment with the following command:

```
source activate usr_intelpython
```

### Install Jupyter Notebook
```
conda install jupyter nb_conda_kernels
```

#### View in Jupyter Notebook

_Note: This distributed execution cannot be launched from the jupyter notebook version, but you can still view inside the notebook to follow the included write-up and description._

Launch Jupyter Notebook in the directory housing the code example

```
jupyter notebook
```

## Running the Sample<a name="running-the-sample"></a>

### Running the Sample as a Python File

When using daal4py for distributed memory systems, the command needed to execute the program should be executed in a bash shell. To execute this example, run the following command, where the number **4** is chosen as an example and means that it will run on **4 processes**:

Run the Program

`mpirun -n 4 python ./IntelPython_daal4py_Distributed_LinearRegression.py`

The output of the script will be saved in the included models and result directories. 

_Note: This code samples focus on using daal4py to do distributed ML computations on chunks of data. The `mpirun` command above will only run on a single local node. To launch on a cluster, you will need to create a host file on the master node, among other steps. The **TensorFlow_Multinode_Training_with_Horovod** code sample explains this process well._

### Running Samples on the Intel&reg; DevCloud (Optional)<a name="run-samples-on-devcloud"></a>

<!---Include the next paragraph ONLY if the sample runs in batch mode-->
### Run in Batch Mode
This sample runs in batch mode, so you must have a script for batch processing. Once you have a script set up, refer to [Running the Sample](#running-the-sample).

### Request a Compute Node
In order to run on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
 This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

<!---Mark each compatible Node in BOLD-->
| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
| __CPU__           | __qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh__      |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |


##### Expected Printed Output (with similar numbers, printed 4 times):
```


Here's our model:


 NumberOfBetas: 15

NumberOfResponses: 1

InterceptFlag: False

Beta: array(
  [[ 0.00000000e+00 -3.20923431e-03 -1.06404233e-01  5.46052700e-02
     2.86834741e-03  2.75997053e+00 -2.54371297e+00  5.52421949e+00
     6.67604639e-04 -9.01293646e-01  1.96091421e-01 -7.50083536e-03
    -3.11567377e-01  1.58333298e-02 -4.62941338e-01]],
  dtype=float64, shape=(1, 15))

NumberOfFeatures: 14

Here is one of our loaded model's features:

 [[ 0.00000000e+00 -3.20923431e-03 -1.06404233e-01  5.46052700e-02
   2.86834741e-03  2.75997053e+00 -2.54371297e+00  5.52421949e+00
   6.67604639e-04 -9.01293646e-01  1.96091421e-01 -7.50083536e-03
  -3.11567377e-01  1.58333298e-02 -4.62941338e-01]]
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]

```

