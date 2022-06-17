# `Jacobi Iterative` Sample

This Sample Demonstrates the number of iterations needed to solve system of Linear Equations using Jacobi Iterative Method.
This Jacobi-iterative sample is implemented using SYCL for Intel CPU and GPU.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 20.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to migrate CUDA to SYCL
| Time to complete                  | 15 minutes

This sample is migrated from NVIDIA CUDA sample, refer to [NVIDIA Sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/jacobiCudaGraphs).

This sample contains four SYCL versions of the same program: 

`sycl_dpct_migrated`         -> It contains DPCT tool migrated code from CUDA code with manual changes for it to work functionally.

| Component 		| Description
|:---			|:---
| Common 		| Helper utility headerfiles
| src 			| DPCT migrated files(.cpp and .h)
| CMakeLists.txt 	| Build file

`sycl_dpct_output`           -> It contains DPCT tool migrated code(with few API's unmigrated) from CUDA code without manual change, hence not functionally working and no 				   build enabled.
| Component 		| Description
|:---			|:---
| Common 		| Helper utility headerfiles
| src 			| DPCT migrated files(.cpp and .h)

`sycl_migrated`              -> It contains manually migrated SYCL code from CUDA code.

| Component 		| Description
|:---			|:---
| src 			| Manually migrated files(.cpp and .h)
| CMakeLists.txt 	| Build file

`sycl_migrated_optimized`    -> It contains manually migrated SYCL code from CUDA code with atomic operations optimization.

| Component 		| Description
|:---			|:---
| src 			| Manually migrated files(.cpp and .h)
| CMakeLists.txt 	| Build file

## Purpose

The Jacobi method is used to find approximate numerical solutions for systems of linear equations of the form Ax = b in numerical linear algebra, which is diagonally dominant. The algorithm starts with an initial estimate for x and iteratively updates it until convergence. The Jacobi method is guaranteed to converge if the matrix A is diagonally dominant.

## Key implementation details

SYCL implementations are explained in the code using key concepts such as Cooperative Groups, Shared Memory, Reduction Stream Capture, and Atomics.

In our case, the matrix is initiated with inputs by generating it randomly with NROWS in createLinearSystem function.
All computations happen inside a for-loop. There are two exit criteria from the loop, first is when we reach maximum number of iteration and second is when the final error falls below the desired tolerance.

Each iteration has two parts: Jacobi Method computation and Final Error computation.

Here we compute the resulting vector of the iteration x_new. Each iteration of the Jacobi method performs the following update for the resulting vector:

```
x_new = D^{-1}(b - (A - D) x)
```

where n x n matrix D is a diagonal component of the matrix A. Vector x is the result of the previous iteration (or an initial guess at the first iteration).  Vector x_new is the result of the current iteration.

In the sample, this computation is offloaded to the `Jacobi Method` device. In both Jacobi method and final error computations we use shared memory, cooperative groups and reduction. x and b vectors are loaded into shared memory for the faster and frequent memory access to the block.

Cooperative groups are used in further dividing the work group into subgroups. Since the computation shown above happens inside subgroups which eliminates the need of block barriers and also are apt for the low granularity of reduction algorithm having each thread run much more efficiently or distributing the work effectively.

The reduction is performed using sync() to synchronize over different thread blocks rather than over entire grid so the implementation is lot more faster avoiding synchronization block.

Shift group left is a SYCL primitive used to do the computation within the subgroup to add all the thread values and are passed on to the first thread. And all the subgroup sums are added through atomic add.

To calculate the `Final error`, we added the absolute value of x substracted with 1 to the warpsum(each thread values are added) and then all the warpsum values are added to the blocksum. And the final error is stored in the g_sum.

At each iteration we compute the final error as:
```
g_sum =  Σ (x - 1)
```


## Building the Jacobi-Iterative Program for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux system wide installations: `. /opt/intel/oneapi/setvars.sh`
>
> Linux private installations: `. ~/intel/oneapi/setvars.sh`
>
>For more information on environment variables, see Use the setvars Script for [Linux](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).


### On a Linux System

Perform the following steps:
1.	Build the program with cmake using the following shell commands. From the root directory of the jacobi project:

	```
	$ mkdir build
	$ cd build
	$ cmake ..
	$ make
	```
	 This builds  `sycl_dpct_migrated`, `sycl_migrated` and `sycl_migrated_optimized` versions of the program.
2.	Run the program:

		Run sycl_dpct_migrated using following commands,
			$ make run_sdm_cpu  (for CPU device)
			$ make run_sdm_gpu  (for GPU device)

		Run sycl_migrated using following commands,
			$ make run_cpu
			$ make run_gpu

		Run sycl_migrated_optimized using following commands,
			$ make run_smo_cpu
			$ make run_smo_gpu

3.	Clean the program using:

	```
	$ make clean
	```

If an error occurs, you can get more details by running make with the VERBOSE=1 argument: make VERBOSE=1 For more comprehensive troubleshooting, use the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors. [Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Example of Output for NROWS = 1024

sycl_dpct_migrated for CPU

	CPU iterations : 6263
	CPU error : 4.987e-03
	CPU Processing time: 1598.485962 (ms)
	GPU iterations : 6263
	GPU error : 4.987e-03
	GPU Processing time: 9653.565430 (ms)
	jacobiSYCL PASSED

sycl_dpct_migrated for GPU

	CPU iterations : 6263
	CPU error : 4.987e-03
	CPU Processing time: 1306.244019 (ms)
	GPU iterations : 6263
	GPU error : 4.987e-03
	GPU Processing time: 4290.418945 (ms)
	jacobiSYCL PASSED

sycl_migrated for CPU

	Serial Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 1560.310547 (ms)

	Running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
	Parallel Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 8629.206055 (ms)
	JacobiSYCL PASSED

sycl_migrated for GPU

	Serial Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 1288.634888 (ms)

	Running on Intel(R) UHD Graphics P630 [0x3e96]
	Parallel Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 4202.227051 (ms)
	JacobiSYCL PASSED

sycl_migrated_optimized for CPU

	Serial Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 1595.077881 (ms)

	Running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
	Parallel Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 9081.323242 (ms)
	JacobiSYCL PASSED

sycl_migrated_optimized for GPU

	Serial Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 1315.337280 (ms)

	Running on Intel(R) UHD Graphics P630 [0x3e96]
	Parallel Implementation :
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 5225.020996 (ms)
	JacobiSYCL PASSED


### Running Samples In DevCloud
If running a sample in the Intel DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

1. Open a terminal on your Linux system.

2. Log in to DevCloud.
	```
	ssh devcloud
	```
3. Download the samples.
	```
	git clone https://github.com/oneapi-src/oneAPI-samples.git
	```

4. Change directories to the Jacobi-Iterative sample directory.
	```
	cd ~/oneAPI-samples/DirectProgramming/DPC++/DenseLinearAlgebra/jacobi-iterative
	```
5. Build the sample on GPU node using
	```
	qsub  -I  -l nodes=1:gpu:ppn=2 -d .
	```
   Note: -I (Upper case I) is used for Interactive mode, -l nodes=1:gpu:ppn=2 (lower case L) is used to assign one full GPU node to the job. Note: The -d . is used to 	          configure the current folder as the working directory for the task.

6. Perform the same steps similar to Linux system.
7. Clean-up the project files
	```
	make clean
	```
8. Disconnect from the Intel DevCloud.
	```
	exit
	```
## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).