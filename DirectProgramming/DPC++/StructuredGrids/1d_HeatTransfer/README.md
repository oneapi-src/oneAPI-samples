# 1D-Heat-Transfer Sample

This code sample demonstrates the simulation of one dimensional heat transfer process using
Intel Data Parallel C++. Kernels in this example are implemented as a discretized differential
equation with second derivative in space and first derivative in time.

For comprehensive instructions regarding DPC++ Programming, go to
https://software.intel.com/en-us/oneapi-programming-guide
and search based on relevant terms noted in the comments.
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to simulate 1D Heat Transfer using Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 10 minutes


## Purpose

1D-Heat-Transfer is a DPC++ application that simulates the heat propagation on a one-dimensional 
isotropic and homogeneous medium. The following equation is used in the simulation of heat propagation:

dU/dt = k * d<sup>2</sup>U/dx<sup>2</sup>

Where:
dU/dt is the rate of change of temperature at a point.
k is the thermal difusivity.
d<sup>2</sup>U/dx<sup>2</sup> is the second spatial derivative.

Or

U(i) = C * (U(i+1) - 2 * U(i) + U(i-1)) + U(i)

where constant C = k * dt / (dx * dx)

The code sample includes both parallel and serial calculation of heat propagation. The code sample will 
attempt first to execute on an available GPU and fallback to the system's CPU if a compatible GPU is 
not detected. The results are stored in a file.


## Key Implementation Details 

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.


## License  

This code sample is licensed under MIT license  


## Building the `1d_HeatTransfer` Program for CPU and GPU

### Include Files  
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system".  

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, 
FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI 
Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System  
  1. Build the program using the following `cmake` commands. 
  
  ```
  $ cd 1d_HeatTransfer
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make -j
  ```

  2. Run the program
  
  ```
  make run  
  ```
  
  3. Clean the program  
  
  ```
  make clean
  ```
  
### On a Windows* System

* Build the program using VS2017 or VS2019
  Right click on the solution file and open using either VS2017 or VS2019 IDE.
  Right click on the project in Solution explorer and select Rebuild.
  From top menu select Debug -> Start without Debugging.

* Build the program using MSBuild
  Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for
 VS2019"
  Run - MSBuild 1d_HeatTransfer.sln /t:Rebuild /p:Configuration="Release"


## Running the sample
### Application Parameters   
	
        Usage: 1d_HeatTransfer <n> <i>

where

n is the number of points you want to simulate the heat transfer.

i is the number of timesteps in the simulation. 

The sample offloads the computation to GPU and also performs a computation in serial in the CPU. 
The results from the parallel and serial computation are compared and stored in an output file named "error_diff.txt". 
If the results match, the application will display a “PASSED” message.  

### Example of Output
```
$ ./1d_HeatTransfer 100 1000
Number of points: 100
Number of iterations: 1000
Kernel runs on Intel(R) Gen9 HD Graphics NEO
Kernel time: 0.347854 sec
PASSED! There is no difference between the results computed in host and in kernel.
```
