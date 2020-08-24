# tbb-async-sycl sample
This sample illustrates how computational kernel can be split for execution between CPU and GPU using TBB Flow Graph asynchronous node and functional node. The Flow Graph asynchronous node uses SYCL to implement calculations on GPU while the functional node does CPU part of calculations. This tbb-async-sycl sample code is implemented using C++ and SYCL language for CPU and GPU.  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta) 
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes

  
## Key implementation details 
TBB Flow Graph and DPC++ implementation explained. 

## License  
This code sample is licensed under MIT license

## How to Build

### On Linux
    * Build tbb-async-sycl program
      cd tbb-async-sycl &&
      mkdir build &&
      cd build &&
      cmake ../. &&
      make VERBOSE=1

    * Run the program
      make run

    * Clean the program
      make clean

### On Windows

#### Command line using MSBuild
     * MSBuild tbb-async-sycl.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE
     * Open Visual Studio 2017
     * Select Menu "File > Open > Project/Solution", find "tbb-async-sycl" folder and select "tbb-async-sycl.sln"
     * Select Menu "Project > Build" to build the selected configuration
     * Select Menu "Debug > Start Without Debugging" to run the program
