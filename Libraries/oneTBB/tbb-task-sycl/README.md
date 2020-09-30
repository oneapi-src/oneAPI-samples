# TBB-Task-Sycl Sample
This sample illustrates how 2 TBB tasks can execute similar computational kernels with one task executing SYCL code and another one the TBB code. This tbb-task-sycl sample code is implemented using C++ and SYCL language for CPU and GPU.
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler (beta) 
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes

## Purpose
The Purpose of this sample is to show how similar computational kernels can be executed by two TBB tasks with one excuting on SYCL code and another on TBB code. 
  
## Key Implementation Details 
The implementation based on TBB tasks and SYCL explained. 

## License  
This code sample is licensed under MIT license   

## Building the TBB-Task-Sycl Program 

### On a Linux System 
    * Build tbb-task-sycl program 
      cd tbb-task-sycl &&
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
     * MSBuild tbb-task-sycl.sln /t:Rebuild /p:Configuration="debug"
   
#### Visual Studio IDE
     * Open Visual Studio 2017
     * Select Menu "File > Open > Project/Solution", find "tbb-task-sycl" folder and select "tbb-task-sycl.sln"
     * Select Menu "Project > Build" to build the selected configuration
     * Select Menu "Debug > Start Without Debugging" to run the program
     
## Running the Sample

### Application Parameters

None

### Example of Output
    executing on CPU
    executing on GPU
    Heterogenous triad correct.
    TBB triad correct.
    input array a_array: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    input array b_array: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    output array c_array on GPU: 0 1.5 3 4.5 6 7.5 9 10.5 12 13.5 15 16.5 18 19.5 21 22.5
    output array c_array_tbb on CPU: 0 1.5 3 4.5 6 7.5 9 10.5 12 13.5 15 16.5 18 19.5 21 22.5
    Built target run
 
