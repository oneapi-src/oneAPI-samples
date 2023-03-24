# `Intel® Python Daal4py Distributed K-Means` Sample

This sample code shows how to train and predict with a distributed k-means model using the Python API package Daal4py powered by the Intel® oneAPI Data Analytics Library (oneDAL).

| Area                  | Description
|:---                   |:---
| What you will learn   | How to use distributed K-Means Daal4py programming model for CPUs
| Time to complete      | 5 minutes
| Category              | Code Optimization

## Purpose

Daal4py is a simplified API to oneDAL that allows for fast usage of the framework suited for Data Scientists or Machine Learning developers. The sample is intended to provide an abstraction to oneDAL for direct usage or integration into your development framework.

In this sample, you will run a distributed K-Means model with oneDAL Daal4py library memory objects. You will also learn how to train a model and save the information to a file.

## Prerequisites

| Optimized for   | Description
|:---             |:---
| OS              | Ubuntu* 18.04 or higher
| Hardware        | Intel Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable processor family
| Software        | Intel® AI Analytics Toolkit (AI Kit)

The sample assumes you have a working version of the Intel® MPI Library, Daal4py, and scikit-learn installed inside a conda environment (similar to what is delivered with the installation of the Intel® Distribution for Python* as part of the AI Kit.)

## Key Implementation Details

This distributed K-means sample code is designed to run on **CPUs**, and is written in Python.

The sample demonstrates how to use software products that are powered by [Intel® oneAPI Data Analytics Library (oneDAL)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onedal.html) and the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/en-us/oneapi/ai-kit).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Intel® Python Daal4py Distributed K-Means` Sample

The Intel® oneAPI Data Analytics Library is ready for use once you finish the Intel® AI Analytics Toolkit installation and have run the post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) for post-installation steps and scripts.

### On Linux*

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

#### Activate Conda with Root Access

By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it. However, if you activated another environment, you can return with the following command.

   ```
   source activate base
   ```

#### Activate Conda without Root Access (Optional)

You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

```
conda create --name usr_intelpython --clone base
source activate usr_intelpython
```

#### Run the Python Script

When using Daal4py for distributed memory systems, you must execute the program in a bash shell.

1. Run the script with a command similar to the following command. (The number **4** is an example and indicates that the script will run on **4 processes**.)
   ```
   mpirun -n 4 python ./IntelPython_daal4py_Distributed_Kmeans.py
   ```
   >**Note**: This code sample uses Daal4py to perform distributed ML computations on chunks of data. The `mpirun` command above will only run on a single local node. To launch on a cluster, you will need to create a host file on the primary node, among other steps. The **TensorFlow_Multinode_Training_with_Horovod** code sample explains this process well.

   When it completes, the script output will be in the included **/models** and **/results** directories.

#### Jupyter Notebook (Optional)

>**Note**: This sample cannot be launched from the Jupyter Notebook version; however, you can still view inside the notebook to follow the included write-up and description.

1. If you have not already done so, install Jupyter Notebook.
   ```
   conda install jupyter nb_conda_kernels
   ```

2. Launch Jupyter Notebook.
   ```
   jupyter notebook
   ```

3. Locate and select the Notebook.
   ```
   IntelPython_daal4py_Distributed_Kmeans.ipynb
   ```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

### Build and Run the Sample on Intel® DevCloud (Optional)

>**Note**: For more information on using Intel® DevCloud, see the Intel® oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started/) page.

1. Open a terminal on a Linux* system.
2. Log in to the Intel® DevCloud.
   ```
   ssh devcloud
   ```
3. If the sample is not already available, download the samples from GitHub.
   ```
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```
4. Change to the sample directory.
5. Configure the sample for the appropriate node.

   The following example is for a CPU node. (This is a single line script.)
	```
	qsub  -I  -l nodes=1:cpu:ppn=2 -d .
	```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:cpu:ppn=2` (lower case L) assigns one full GPU node.
   - `-d .` makes the current folder as the working directory for the task.

     >**Note**: For more information about the node properties, execute the `pbsnodes` command.

6. Perform build steps you would on Linux.
7. Run the sample.

> **Note**: To inspect job progress if you are using a script, use the qstat utility.
>   ```
>   watch -n 1 qstat -n -1
>   ```
>  The command displays the results every second. The job is complete when no new results display.

8. Review the output.
9. Disconnect from Intel® DevCloud.
	```
	exit
	```

## Example Output

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

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).