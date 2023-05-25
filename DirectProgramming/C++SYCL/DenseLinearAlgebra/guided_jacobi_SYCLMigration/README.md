# `Jacobi Iterative` Sample
The `Jacobi Iterative` sample demonstrates the number of iterations needed to solve system of Linear Equations using Jacobi Iterative Method. This sample is implemented using SYCL* by migrating code from original CUDA source.

> **Note**: This sample is migrated from NVIDIA CUDA sample. See the [jacobiCudaGraphs](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/jacobiCudaGraphs) sample in the NVIDIA/cuda-samples GitHub.

| Property                          | Description
|:---                               |:---
| What you will learn               | How to begin migrating CUDA to SYCL
| Time to complete                  | 15 minutes

## Purpose
The Jacobi method is used to find approximate numerical solutions for systems of linear equations of the form $Ax = b$ in numerical linear algebra, which is diagonally dominant. The algorithm starts with an initial estimate for x and iteratively updates it until convergence. The Jacobi method is guaranteed to converge if the matrix A is diagonally dominant.

This [Migrating the Jacobi Iterative Method from CUDA* to SYCL*](https://www.intel.com/content/www/us/en/developer/articles/technical/cuda-sycl-migration-jacobi-iterative-method.html) article provides a detailed explain on how the migration from CUDA to SYCL:
- You can learn about how the original CUDA code is migrated to SYCL using Intel® DPC++ Compatibility Tool and how to address the warnings generated in the output.
- You will learn how to analyze the CUDA source step-by-step and manually migrate to SYCL by replacing CUDA calls with equivalent SYCL calls.
- You will learn how to performance optimize the SYCL code.

This sample contains four versions:

|Folder Name   |Description
|:---          |:---
|`01_sycl_dpct_output`	| Contains output of Intel® DPC++ Compatibility Tool used to migrate SYCL-compliant code from CUDA code, this SYCL code has some unmigrated code which has to be manually fixed to get full functionality, the code does not functionally work.
|`02_sycl_dpct_migrated`	| Contains Intel® DPC++ Compatibility Tool migrated SYCL code from CUDA code with manual changes done to fix the unmigrated code to work functionally.
|`03_sycl_migrated`	| Contains manually migrated SYCL code from CUDA code (without using Intel® DPC++ Compatibility Tool).
|`04_sycl_migrated_optimized`	| Contains manually migrated SYCL code from CUDA code with performance optimizations applied.

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The matrix is initiated with inputs by generating it randomly with `NROWS` in createLinearSystem function. All computations happen inside a for-loop. There are two exit criteria from the loop, first is when we reach maximum number of iterations and second is when the final error falls below the desired tolerance. Each iteration has two parts:
- Jacobi Method computation
- Final Error computation

The loops compute the resulting vector of the iteration x_new. Each iteration of the Jacobi method performs the following update for the resulting vector:

```
x_new = $D^{-1}(b - (A - D) x)$
```

where:
- n x n matrix D is a diagonal component of the matrix A.
- Vector x is the result of the previous iteration (or an initial guess at the first iteration).
- Vector x_new is the result of the current iteration.

Key SYCL concepts explained in the code are Cooperative Groups, Shared Memory, Reduction Stream Capture, and Atomics.

In the sample, this computation is offloaded to the `Jacobi Method` device. In both `Jacobi Method` and `Final Error` computations we use shared memory, cooperative groups and reduction. x and b vectors are loaded into shared memory for the faster and frequent memory access to the block.

Cooperative groups are used in further dividing the work group into subgroups. Since the computation shown above happens inside subgroups, which eliminates the need of block barriers and are apt for the low granularity of reduction algorithm having each thread run much more efficiently or distributing the work effectively.

The reduction is performed using `sync()` to synchronize over different thread blocks rather than over entire grid so the implementation is faster avoiding synchronization block.

Shift group left is a SYCL primitive used to do the computation within the subgroup to add all the thread values and are passed on to the first thread. And all the subgroup sums are added through atomic add.

To calculate the `Final error`, we added the absolute value of x minus 1 to the warpsum (each thread value is added) and then all the warpsum values are added to the blocksum. And the final error is stored in the g_sum.

In each iteration, we compute the final error as:
```
g_sum =  Σ (x - 1)
```

## Build the `Jacobi Iterative` Sample for CPU and GPU
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `source . /opt/intel/oneapi/setvars.sh`
> - For private installations: `source . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system. You might need to use some of the resources from this location to build the sample.

### On Linux*
Perform the following steps:
1. Change to the `jacobi-iterative` directory.
2. Build the program. 
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
   By default, these commands build the `sycl_dpct_migrated`, `sycl_migrated` and `sycl_migrated_optimized` versions of the program.

If an error occurs, you can get more details by running make with the VERBOSE=1 argument: 
```
make VERBOSE=1
```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Jacobi Iterative` Sample
In all cases, you can run the programs for CPU and GPU. The run commands indicate the device target.

1. Run `02_sycl_dpct_migrated` for GPU.
   ```
   make run_sdm
   ```
   Run `02_sycl_dpct_migrated` for CPU.
   ```
   export SYCL_DEVICE_FILTER=cpu
   make run_sdm
   unset SYCL_DEVICE_FILTER
   ```

2. Run `03_sycl_migrated` for GPU.
   ```
   make run
   ```
   Run `03_sycl_migrated` for CPU.
   ```
   export SYCL_DEVICE_FILTER=cpu
   make run
   unset SYCL_DEVICE_FILTER
   ```
3. Run `04_sycl_migrated_optimized` for GPU.
   ```
   make run_smo
   ```
   Run `04_sycl_migrated_optimized` for CPU.
   ```
   export SYCL_DEVICE_FILTER=cpu
   make run_smo
   unset SYCL_DEVICE_FILTER
   ```

### Run the `Jacobi Iterative` Sample In Intel® DevCloud
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
4. Change to the `jacobi-iterative` directory.
	```
	cd ~/oneAPI-samples/DirectProgramming/DPC++/DenseLinearAlgebra/jacobi-iterative
	```
5. Configure the sample for a GPU node using `qsub`. 
	```
	qsub  -I  -l nodes=1:gpu:ppn=2 -d .
	```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
   - `-d .` makes the current folder as the working directory for the task.
6. Perform build steps as you would on Linux.
7. Run the sample.
8. Clean up the project files.
	```
	make clean
	```
9. Disconnect from the Intel® DevCloud.
	```
	exit
	```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
