# dpc_reduce Sample

The dpc_reduce is a simple program that calculates pi.  This program is implemented using C++ and Data Parallel C++ (DPC++) for Intel(R) CPU and accelerators. This code sample also demonstrates how to incorporate Data Parallel PC++ into
a MPI program.


For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS	                | Linux* Ubuntu* 18.04, 
| Hardware	            | Skylake with GEN9 or newer, 
| Software	            | Intel® oneAPI DPC++ Compiler (beta)
| What you will learn   | how to perform reduction with oneAPI on cpu and gpu
| Time to complete      | 30 min 

## Purpose
This example demonstrates how to do reduction by using the CPU in serial mode, 
the CPU in parallel mode (using TBB), the GPU using direct DPC++ coding, the 
GPU using multiple steps with DPC++ Library algorithms transform and reduce, 
and then finally using the DPC++ Library transform_reduce algorithm.  

All the different modes use a simple calculation for Pi.   It is a well known 
mathematical formula that if you integrate from 0 to 1 over the function, 
(4.0 / (1+x*x) )dx the answer is pi.   One can approximate this integral 
by summing up the area of a large number of rectangles over this same range.  

Each of the different function calculates pi by breaking the range into many 
tiny rectangles and then summing up the results. 

The parallel computations are performed using oneTBB and oneAPI DPC++ library 
(oneDPL).

This example also demonstrates how to incorporate Data Parallel PC++ into a MPI program.
Using Data Parallel C++, the code sample runs multiple MPI ranks to distribute the
calculation of the number Pi. Each rank offloads the computation to an accelerator
(GPU/CPU) using Intel DPC++ compiler to compute a partial compution of the number Pi.

If you run the sample on a CPU as your default device,  you may need to increase 
the memory allocation for openCL.  You can do this by setting an environment variable, 
    "CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=16MB


## Key Implementation Details
The basic DPC++ implementation explained in the code includes accessor,
kernels, queues, buffers as well as some oneDPL library calls. 

## License
This code sample is licensed under MIT license.

## Building the dpc_reduce program for CPU and GPU

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system".

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
NOTE:  Location of some include files changed between Beta09 and Beta10. 
See the CMakeLists.txt and uncomment out line if you want Beta09 to build. 

Perform the following steps:
1. Build the program using the following 'cmake' commands
export I_MPI_CXX=dpcpp 
mkdir build 
cd build 
cmake .. 
make 

2. Run the program using:
make run or './dpc_reduce' or 'mpirun ./dpc_reduce'

3. Clean the program using:
make clean


## Running the Sample
### Application Parameters

        Usage: mpirun -n <num> ./dpc_reduce

where

        <num> : number of MPI rank.


### Example of Output
```c++
Rank #0 runs on: lqnguyen-NUC1, uses device: Intel(R) Gen9 HD Graphics NEO \
Number of steps is 1000000 \
Cpu Seq calc:           PI =3.14 in 0.00422 seconds \
Cpu TBB  calc:          PI =3.14 in 0.00177 seconds \
dpstd native:           PI =3.14 in 0.209 seconds \
dpstd native2:          PI =3.14 in 0.213 seconds \
dpstd native3:          PI =3.14 in 0.00222 seconds \
dpstd native4:          PI =3.14 in 0.00237 seconds \
dpstd two steps:        PI =3.14 in 0.0014 seconds \
dpstd transform_reduce: PI =3.14 in 0.000528 seconds \
mpi native:             PI =3.14 in 0.548 seconds \
mpi transform_reduce:   PI =3.14 in 0.000498 seconds \
succes \
```
