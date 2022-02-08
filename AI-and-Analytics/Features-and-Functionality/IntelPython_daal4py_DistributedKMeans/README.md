# `Intel Python daal4py Distributed K-Means` Sample
This sample code shows how to train and predict with a distributed k-means model using the python API package daal4py for oneAPI Data Analytics Library. It assumes you have a working version of the Intel MPI library installed, and it demonstrates how to use software products that can be found in the [oneAPI Data Analytics Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) or [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher, 64-bit Windows 10, macOS 10.14 or higher
| Hardware                          | Intel Atom® Processors; Intel® Core™ Processor Family; Intel® Xeon® Processor Family; Intel® Xeon® Scalable  processor family
| Software                          | oneAPI Data Analytics Library (oneDAL), Python version >= 3.6, conda-build version >= 3, C++ compiler with C++11 support, Pickle, Pandas, NumPy
| What you will learn               | distributed oneDAL K-Means programming model for Intel CPU
| Time to complete                  | 5 minutes

## Purpose

daal4py is a simplified API to Intel® oneDAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning users. Built to help provide an abstraction to Intel® oneDAL for direct usage or integration into one's own framework.

In this sample, you will run a distributed K-Means model with oneDAL daal4py library memory objects. You will also learn how to train a model and save the information to a file.

## Key Implementation Details
This distributed K-means sample code is implemented for CPU using the Python language. The example assumes you have daal4py and scikit-learn installed inside a conda environment, similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/en-us/oneapi/ai-kit).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Running Samples on the Intel&reg; DevCloud
If you are running this sample on the DevCloud, see [Running Samples on the Intel&reg; DevCloud](#run-samples-on-devcloud)

## Building daal4py for CPU

oneAPI Data Analytics Library is ready for use once you finish the Intel® oneAPI AI Analytics Toolkit installation and have run the post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) for post-installation steps and scripts.


### Setting Environment Variables

For working at a Command-Line Interface (CLI), the tools in the oneAPI toolkits
are configured using environment variables. Set up your CLI environment by
sourcing the ``setvars`` script every time you open a new terminal window. This
will ensure that your compiler, libraries, and tools are ready for development.

#### Linux

Source the script from the installation location, which is typically in one of
these folders:

For root or sudo installations:


  ``. /opt/intel/oneapi/setvars.sh``


For normal user installations:

  ``. ~/intel/oneapi/setvars.sh``

**Note:** If you are using a non-POSIX shell, such as csh, use the following command:

     ``$ bash -c 'source <install-dir>/setvars.sh ; exec csh'``

If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


**Note:** [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html)
    can also be used to set up your development environment.
    The modulefiles scripts work with all Linux shells.


**Note:** If you wish to fine
    tune the list of components and the version of those components, use
    a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html)
    to set up your development environment.


#### Windows

Execute the  ``setvars.bat``  script from the root folder of your
oneAPI installation, which is typically:


  ``"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"``


For Windows PowerShell* users, execute this command:

  ``cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'``


If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### Activate conda environment With Root Access

Intel Python environment will be active by default. However, if you activated another environment, you can return with the following command:

#### On a Linux* System
```
source activate base
```

### Activate conda environment Without Root Access (Optional)

By default, the Intel® oneAPI AI Analytics toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

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

### Running the Sample as a Python File<a name="running-the-sample"></a>

When using daal4py for distributed memory systems, the command needed to execute the program should be executed in a bash shell. To execute this example, run the following command, where the number **4** is chosen as an example and means that it will run on **4 processes**:

Run the Program

`mpirun -n 4 python ./IntelPython_daal4py_Distributed_Kmeans.py`

The output of the script will be saved in the included models and result directories.

_Note: This code samples focus on using daal4py to do distributed ML computations on chunks of data. The `mpirun` command above will only run on a single local node. To launch on a cluster, you will need to create a host file on the master node, among other steps. The **TensorFlow_Multinode_Training_with_Horovod** code sample explains this process well._

### Using Visual Studio Code* (VS Code)

You can use VS Code extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Running Samples on the Intel&reg; DevCloud (Optional)<a name="run-samples-on-devcloud"></a>

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


Here our centroids:


 [[ 5.46000000e+02 -3.26170648e+00 -6.15922494e+00]
 [ 1.80000000e+01 -1.00432059e+01 -8.38198798e+00]
 [ 4.10000000e+02  3.78330964e-01  8.29073839e+00]]

Here is our centroids loaded from file:

 [[ 5.46000000e+02 -3.26170648e+00 -6.15922494e+00]
 [ 1.80000000e+01 -1.00432059e+01 -8.38198798e+00]
 [ 4.10000000e+02  3.78330964e-01  8.29073839e+00]]
Here is our cluster assignments for first 5 datapoints:

 [[1]
 [1]
 [1]
 [1]
 [1]]
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]

```


