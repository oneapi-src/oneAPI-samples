# TBB-Async-Sycl Sample
This sample illustrates how computational kernel can be split for execution between CPU and GPU using TBB Flow Graph asynchronous node and functional node. The Flow Graph asynchronous node uses SYCL to implement calculations on GPU while the functional node does CPU part of calculations. This tbb-async-sycl sample code is implemented using C++ and SYCL language for CPU and GPU.  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler (beta) 
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes

## Purpose
The purpose of this sample is to show how during execution, a computational kernel can be split be between CPU and GPU using TBB Flow Graph asynchronous node and functional node.

## Key Implementation Details 
TBB Flow Graph and DPC++ implementation explained. 

## License  
This code sample is licensed under MIT license

## Building the TBB-Async-Sycl Program

### On a Linux System
    * Build tbb-async-sycl program
      cd tbb-async-sycl &&
      mkdir build &&
      cd build &&
      cmake .. &&
      make VERBOSE=1

    * Run the program
      make run

    * Clean the program
      make clean

### On a Windows System

#### Command line using MSBuild
     * MSBuild tbb-async-sycl.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE
     * Open Visual Studio 2017
     * Select Menu "File > Open > Project/Solution", find "tbb-async-sycl" folder and select "tbb-async-sycl.sln"
     * Select Menu "Project > Build" to build the selected configuration
     * Select Menu "Debug > Start Without Debugging" to run the program
     
## Running the Sample

### Application Parameters
None

### Example of Output

    start index for GPU = 0; end index for GPU = 8
    start index for CPU = 8; end index for CPU = 16
    Heterogenous triad correct.
    c_array: 0 1.5 3 4.5 6 7.5 9 10.5 12 13.5 15 16.5 18 19.5 21 22.5
    c_gold : 0 1.5 3 4.5 6 7.5 9 10.5 12 13.5 15 16.5 18 19.5 21 22.5
    Built target run

     
    
