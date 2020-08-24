# openmp_reduction Sample

The openmp_reduction sample is a simple program that calculates pi.  This program is implemented using C++ and openMP for Intel(R) CPU and accelerators.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.


| Optimized for                     | Description
|:---                               |:--- 
| OS	                  | Linux* Ubuntu* 18.04, 
| Hardware	            | Skylake with GEN9 or newer
| Software	            | Intel® oneAPI DPC++ Compiler (beta)
| What you will learn   | How to run openMP on cpu as well as GPU offload 
| Time to complete      | 10 min

## Purpose 
This example demonstrates how to do reduction by using the CPU in serial mode, 
the CPU in parallel mode (using openMP), the GPU using openMP offloading. 

All the different modes use a simple calculation for Pi.   It is a well known 
mathematical formula that if you integrate from 0 to 1 over the function, 
(4.0 / (1+x*x) )dx the answer is pi.   One can approximate this integral 
by summing up the area of a large number of rectangles over this same range.  

Each of the different functions calculates pi by breaking the range into many 
tiny rectangles and then summing up the results. 

## Key Implementation Details
This code shows how to use OpenMP on the CPU host as well as using target offload capabilities. 

## License
This code sample is licensed under MIT license.

## Building the dpc_reduce program for CPU and GPU

### Include Files  
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system".  

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:

mkdir build 
cd build 
cmake .. 

1.  Build the program using the following make commands 
make 

2. Run the program using:
make run or src/openmp_reduction 

3.  Clean the program using:
make clean


## Running the Sample

### Application Parameters
There are no editable parameters for this sample.

### Example of Output (result vary depending on hardware)
Number of steps is 1000000

Cpu Seq calc:           PI =3.14 in 0.00105 seconds

Host OpenMP:            PI =3.14 in 0.0010 seconds

Offload OpenMP:         PI =3.14 in 0.0005 seconds

success
