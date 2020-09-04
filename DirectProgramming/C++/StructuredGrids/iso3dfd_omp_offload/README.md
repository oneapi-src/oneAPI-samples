# `ISO3DFD OpenMP Offload` Sample

The ISO3DFD sample refers to Three-Dimensional Finite-Difference Wave Propagation in Isotropic Media.  It is a three-dimensional stencil to simulate a wave propagating in a 3D isotropic medium and shows some of the more common challenges and techniques when targeting OMP Offload devices (GPU) in more complex applications to achieve good performance. 

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler;
| What you will learn               | How to offload the computation to GPU using Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

Performance number tabulation

| iso3dfd_omp_offload sample            | Performance data
|:---                               	|:---
| Default Baseline version              | 1.0
| Optimized version 1	                | 1.11x
| Optimized version 2	                | 1.48x
| Optimized version 3	                | 1.60x


## Purpose

ISO3DFD is a finite difference stencil kernel for solving the 3D acoustic isotropic wave equation which can be used as a proxy for propogating a seismic wave. Kernels in this sample are implemented as 16th order in space, with symmetric coefficients, and 2nd order in time scheme without boundary conditions.. Using OpenMP Offload, the sample can explicitly run on the GPU to propagate a seismic wave which is a compute intensive task.

The code will attempt to find an available GPU or OpenMP Offload capable device and exit if a compatible device is not detected. By default, the output will print the device name where the OpenMP Offload code ran along with the grid computation metrics - flops and effective throughput. For validating results, a OpenMP/CPU-only version of the application will be run on host/CPU and results will be compared to the OpenMP Offload version.

The code also demonstrates some of the common optimization techniques which can be used to improve performance of 3D-stencil code running on a GPU device.
 
## Key Implementation Details 

The basic OpenMP Offload implementation explained in the code includes the use of the following : 
* OpenMP offload target data map construct
* Default Baseline version demonstrates use of OpenMP offload target parallel for construct with collapse 
* Optimized version 1 demonstrates use of OpenMP offload teams distribute construct and use of num_teams and thread_limit clause
* Incremental Optimized version 2 demonstrates use of OpenMP offload teams distribute construct with improved data-access pattern
* Incremental Optimized version 3 demonstrates use of OpenMP CPU threads along with OpenMP offload target construct

 
## License  

This code sample is licensed under MIT license. 


## Building the `ISO3DFD` Program for GPU

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/) and Intel® oneAPI HPC Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/hpc-toolkit/)

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands. 
``` 
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```

> Note: by default, executable is build with default baseline version. You can build the kernel with optimized versions with the following:
```
cmake -DUSE_OPT1=1 ..
make -j
```
```
cmake -DUSE_OPT2=1 ..
make -j
```
```
cmake -DUSE_OPT3=1 ..
make -j
```

2. Run the program :
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```

## Running the Sample
```
make run
```

### Application Parameters 
You can modify the ISO3DFD parameters from the command line.
   * Configurable Application Parameters   
	
	Usage: src/iso3dfd n1 n2 n3 n1_block n2_block n3_block Iterations

 	n1 n2 n3                       	: Grid sizes for the stencil
 	n1_block n2_block n3_block     	: cache block sizes for CPU
                                	: OR TILE sizes for OMP Offload
 	Iterations                     	: No. of timesteps.

### Example of Output with default baseline version
```
Grid Sizes: 256 256 256
Tile sizes ignored for OMP Offload
--Using Baseline version with omp target with collapse
Memory Usage (MBytes): 230
--------------------------------------
time         : 4.827 secs
throughput   : 347.57 Mpts/s
flops        : 21.2018 GFlops
bytes        : 4.17084 GBytes/s

--------------------------------------

--------------------------------------
Checking Results ...
Final wavefields from OMP Offload device and CPU are equivalent: Success
--------------------------------------
```

### Example of Output with Optimized version 3
```
Grid Sizes: 256 256 256
Tile sizes: 16 8 64
Using Optimized target code - version 3:
--OMP Threads + OMP_Offload with Tiling and Z Window
Memory Usage (MBytes): 230
--------------------------------------
time         : 3.014 secs
throughput   : 556.643 Mpts/s
flops        : 33.9552 GFlops
bytes        : 6.67971 GBytes/s

--------------------------------------

--------------------------------------
Checking Results ...
Final wavefields from OMP Offload device and CPU are equivalent: Success

```
