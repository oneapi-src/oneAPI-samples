# `odd-Even MergeSort` Sample

This sample implements odd-even merge sort (also known as Batcher's sort)algorithm belonging to the class of sorting networks. While generally subefficient, for large sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort or radix sort), this may be the preferred algorithms of choice for sorting batches of short-sized to mid-sized (key, value) array pairs.

This odd-even merge sort sample is implemented using DPC++ and SYCL for Intel CPU and GPU.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 20.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to migrate CUDA to SYCL
| Time to complete                  | 15 minutes

This Sample is migrated from NVIDIA CUDA sample, Refer [NVIDIA Sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks).

This sample contains three SYCL versions of the same program: 

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
				
`sycl_migrated`              -> It contains Manually migrated SYCL code from CUDA code.

| Component 		| Description
|:---			|:---
| src 			| Manually migrated files(.cpp and .h)
| CMakeLists.txt 	| Build file
			

## Purpose

The odd-even mergesort algorithm was developed by K.E. Batcher. It is based on a merge algorithm that merges two sorted halves of a sequence to a completely sorted sequence.

In contrast to mergesort, this algorithm is not data-dependent, i.e. the same comparisons are performed regardless of the actual data. Therefore, odd-even mergesort can be implemented as a sorting network.


## Key implementation details

DPC++ and SYCL Implementation is explained in the code using key concepts such as Cooperative Groups, Shared Memory and Data-Parallel algorithm.




## Building the odd-even merge sort Program for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
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
	 This builds  `sycl_dpct_migrated`, `sycl_migrated` versions of the program.
	
2.	Run the program:

		Run sycl_dpct_migrated using following commands,
			$ make run_sdm_cpu  (for CPU device)
			$ make run_sdm_gpu  (for GPU device)

		Run sycl_migrated using following commands,
			$ make run_cpu
			$ make run_gpu
			
3.	Clean the program using:

	```
	$ make clean
	```
	

If an error occurs, you can get more details by running make with the VERBOSE=1 argument: make VERBOSE=1 For more comprehensive troubleshooting, use the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors. [Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

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

If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the [Intel® oneAPI Base Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

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
	cd ~/oneAPI-samples/DirectProgramming/DPC++/GraphTraversal/odd-even-merge-sort
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

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
