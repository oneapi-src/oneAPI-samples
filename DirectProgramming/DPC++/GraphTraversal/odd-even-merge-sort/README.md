# `odd-Even MergeSort` Sample

This sample implements odd-even mergesort (also known as Batcher's sort) algorithm belonging to the class of sorting networks. While generally subefficient, for large sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort or radix sort), this may be the preferred algorithms of choice for sorting batches of short-sized to mid-sized (key, value) array pairs.

This odd-even mergesort sample is implemented using DPC++ and SYCL for Intel CPU and GPU.


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
| Common 		| Helper utility headerfiles
| src 			| Manually migrated files(.cpp and .h)
| CMakeLists.txt 	| Build file
			

## Purpose

The odd-even mergesort algorithm was developed by K.E. Batcher. It is based on a merge algorithm that merges two sorted halves of a sequence to a completely sorted sequence.

In contrast to mergesort, this algorithm is not data-dependent, i.e. the same comparisons are performed regardless of the actual data. Therefore, odd-even mergesort can be implemented as a sorting network.


## Key implementation details

DPC++ and SYCL Implementation is explained in the code using key concepts such as Cooperative Groups, Shared Memory and Data-Parallel algorithm.




## Building the odd-even mergesort Program for CPU and GPU

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
1.	Build the program with cmake using the following shell commands. From the root directory of the odd-even-merge-sort project:

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

## Example of Output for array length=1048576
	
sycl_migrated for GPU

	Running on Intel(R) UHD Graphics P630 [0x3e96]
	
	Running GPU oddEvenMerge sort (1 identical iterations)...

	Testing array length 64 (16384 arrays per batch)...
	Average time: 146.904007 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: OK
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 128 (8192 arrays per batch)...
	Average time: 5.515000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: OK
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 256 (4096 arrays per batch)...
	Average time: 6.579000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: OK
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 512 (2048 arrays per batch)...
	Average time: 7.458000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: OK
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 1024 (1024 arrays per batch)...
	Average time: 9.548000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 2048 (512 arrays per batch)...
	Average time: 9.632000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 4096 (256 arrays per batch)...
	Average time: 9.815000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 8192 (128 arrays per batch)...
	Average time: 10.106000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 16384 (64 arrays per batch)...
	Average time: 10.378000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 32768 (32 arrays per batch)...
	Average time: 10.603000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 65536 (16 arrays per batch)...
	Average time: 11.103000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 131072 (8 arrays per batch)...
	Average time: 11.221000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 262144 (4 arrays per batch)...
	Average time: 11.764000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 524288 (2 arrays per batch)...
	Average time: 11.892000 ms


	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Testing array length 1048576 (1 arrays per batch)...
	Average time: 12.206000 ms

	sortingNetworks-oddevenMerge sort, Throughput = 85.9066 MElements/s, Time = 0.01221 s, Size = 1048576 elements, NumDevsUsed = 1, Workgroup = 256

	Validating the results...
	...reading back GPU results
	...inspecting keys array: ***Set 0 result key array is not ordered properly***
	...inspecting keys and values array: OK
	...stability property: NOT stable

	Shutting down...


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

4. Change directories to the odd-even-merge-sort sample directory.
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
