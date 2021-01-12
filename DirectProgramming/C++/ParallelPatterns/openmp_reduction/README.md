# `openmp_reduction` Sample

The `openmp_reduction` code sample is a simple program that calculates pi.  This program is implemented using C++ and openMP for Intel(R) CPU and accelerators.

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.


| Optimized for                     | Description
|:---                               |:--- 
| OS	                  | Linux* Ubuntu* 18.04, 
| Hardware	            | Skylake with GEN9 or newer
| Software	            | Intel® oneAPI DPC++ Compiler
| What you will learn   | How to run openMP on cpu as well as GPU offload 
| Time to complete      | 10 min

## Purpose 
This example demonstrates how to do reduction by using the CPU in serial mode, the CPU in parallel mode (using OpenMP), the GPU using OpenMP offloading. 

All the different modes use a simple calculation for Pi. It is a well known mathematical formula that if you integrate from 0 to 1 over the function, (4.0 / (1+x*x) )dx, the answer is pi. One can approximate this integral by summing up the area of a large number of rectangles over this same range.  

Each of the different functions calculates pi by breaking the range into many tiny rectangles and then summing up the results. 

## Key Implementation Details
This code shows how to use OpenMP on the CPU host as well as using target offload capabilities. 

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the dpc_reduce program for CPU and GPU

### Include Files  
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system".  

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the [Intel® oneAPI Base Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:
```
mkdir build 
cd build 
cmake ..
```
Build the program using the following make commands 
```
make
```
Run the program using:
```
make run
``` 
    or 
```
src/openmp_reduction
``` 
Clean the program using:
```
make clean
```

## Running the Sample

### Application Parameters
There are no editable parameters for this sample.

### Example of Output (result vary depending on hardware)

```
Number of steps is 1000000

Cpu Seq calc:           PI =3.14 in 0.00105 seconds

Host OpenMP:            PI =3.14 in 0.0010 seconds

Offload OpenMP:         PI =3.14 in 0.0005 seconds

success
```