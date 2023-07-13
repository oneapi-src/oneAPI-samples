# `Concurrent Kernels` Sample

The `Concurrent Kernels` sample demonstrates the use of SYCL queues for concurrent execution of several kernels on GPU device. It is implemented using SYCL* by migrating code from original CUDA source code for offloading computations to a GPU or CPU and further demonstrates how to optimize and improve processing time.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to begin migrating CUDA to SYCL
| Time to complete       | 15 minutes

>**Note**: This sample is based on the [concurrentKernels](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/concurrentKernels) sample in the NVIDIA/cuda-samples GitHub repository.

## Purpose

The `Concurrent Kernels` sample shows the execution of multiple kernels on the device at the same time.

This sample contains three versions of the code in the following folders:

| Folder Name             | Description
|:---                     |:---
| `01_sycl_dpct_output`   | Contains output of Intel® DPC++ Compatibility Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_dpct_migrated` | Contains Intel® DPC++ Compatibility Tool migrated SYCL code from CUDA code with manual changes done to fix the unmigrated code to work functionally.
| `03_sycl_migrated`      | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04 (or newer)
| Hardware              | GEN9 (or newer)
| Software              | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

concurrentKernels involves a kernel that does no real work but runs at least for a specified number of iterations.

>**Note**: This sample demonstrates the CUDA concurrentKernels using key concepts such as CUDA streams and Performance Strategies.

SYCL has two kinds of queues that a programmer can create and use to submit kernels for execution.

The choice to create an in-order or out-of-order queue is made at queue construction time through the property sycl::property::queue::in_order(). By default, when no property is specified, the queue is out-of-order.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Concurrent Kernels` Sample

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

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence will build the `02_sycl_dpct_migrated`, `03_sycl_migrated` versions of the program.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `Concurrent Kernels` Sample

### On Linux

You can run the programs for CPU or GPU. The commands indicate the device target.

1. Run `02_sycl_dpct_migrated` for **GPU**.
   ```
   make run_sdm
   ```
   Run `02_sycl_dpct_migrated` for **CPU**.
   ```
   export SYCL_DEVICE_FILTER=cpu
   make run_sdm
   unset SYCL_DEVICE_FILTER
   ```

2. Run `03_sycl_migrated` for **GPU**.
   ```
   make run
   ```
   Run `03_sycl_migrated` for **CPU**.
   ```
   export SYCL_DEVICE_FILTER=cpu
   make run
   unset SYCL_DEVICE_FILTER
   ```

### Build and Run the Sample in Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)

You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`.

1. Open a terminal on a Linux* system.
2. Log in to Intel® DevCloud.
    ```
    ssh devcloud
    ```
3. Download the samples.
    ```
    git clone https://github.com/oneapi-src/oneAPI-samples.git
    ```
4. Change to the sample directory.
5. Configure the sample for a GPU node and choose the backend as OpenCL.
   ```
   qsub  -I  -l nodes=1:gpu:ppn=2 -d .
   export SYCL_DEVICE_FILTER=opencl:gpu
   ```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
   - `-d .` makes the current folder as the working directory for the task.

     |Available Nodes  |Command Options
     |:---             |:---
     | GPU	         |`qsub -l nodes=1:gpu:ppn=2 -d .`
     | CPU	         |`qsub -l nodes=1:xeon:ppn=2 -d .`

6. Perform build steps as you would on Linux.
7. Run the programs.
8. Clean up the project files.
    ```
    make clean
    ```
9. Disconnect from the Intel® DevCloud.
    ```
    exit
    ```

## Example Output

The following example is for `03_sycl_migrated` on **Intel(R) UHD Graphics P630 \[0x3e96\]** with OpenCL Backend.

```
[./a.out] - Starting...
Device: Intel(R) UHD Graphics [0x9a60]
> Detected Compute SM 3.0 hardware with 32 multi-processors
Expected time for serial execution of 8 kernels = 0.080s
Expected time for concurrent execution of 8 kernels = 0.010s
Measured time for sample = 0.07301s
Test passed

```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).