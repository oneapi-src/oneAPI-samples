# `Jacobi Iterative` Sample

This Sample Demonstrates the number of iterations needed to solve system of Linear Equations using Jacobi Iterative Method. 
This Jacobi-iterative sample is implemented using DPC++ and SYCL for Intel CPU and GPU.
	

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 20.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to migrate CUDA to SYCL
| Time to complete                  | 15 minutes

	
## Purpose

The Jacobi method is used to find approximate numerical solutions for systems of linear equations of the form Ax = b in numerical linear algebra, which is diagonally dominant. The algorithm starts with an initial estimate for x and iteratively updates it until convergence. The Jacobi method is guaranteed to converge if the matrix A is diagonally dominant.


## Key implementation details

DPC++ and SYCL Implementation is explained in the code using key concepts such as Stream Capture, Atomics and Cooperative Groups.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the Jacobi-Iterative Program for CPU and GPU

Note: If you have not already done so, set up your CLI environment by sourcing the setvars script located in the root of your oneAPI installation.
Linux Sudo: . /opt/intel/oneapi/setvars.sh	
For more information on environment variables, see Use the setvars Script for Linux or macOS, or Windows

### On a Linux System

Perform the following steps:
1.	Build the program with cmake using the following shell commands. From the root directory of the jacobi project:

	```
	$ mkdir build
	$ cd build
	$ cmake ..
	$ make
	```
	
2.	Run the program:

		For sycl_migrated, run using following steps
			$make run_cpu  (for CPU device)
			$make run_gpu  (for GPU device)

		For sycl_migrated_optimized
			$make run_smo_cpu
			$make run_smo_gpu
			
		For dpct_sycl_migrated
			$make run_dsm_cpu
			$make run_dsm_gpu

3.	Clean the program using:

	```
	$ make clean
	```
	

If an error occurs, you can get more details by running make with the VERBOSE=1 argument: make VERBOSE=1 For more comprehensive troubleshooting, use the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors. Learn more.

Example of Output

	Serial Implementation : 
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 1596.404175 (ms)
	
	Running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
	Parallel Implementation : 
	Iterations : 6263
	Error : 4.987e-03
	Processing time : 8463.482422 (ms)
	JacobiSYCL PASSED


### Running Samples In DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide

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
Note: -I (Upper case I) is used for Interactive mode, -l nodes=1:gpu:ppn=2 (lower case L) is used to assign one full GPU node to the job. Note: The -d . is used to 	       configure the current folder as the working directory for the task.

6. Perform the same steps similar to Linux system.
	
7. Clean-up the project files
	```	
	make clean
	```
	    
8. Disconnect from the Intel DevCloud.
	```
	exit
	```
	
