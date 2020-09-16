# tbb-async-sycl sample
This sample illustrates how computational kernel can be split for execution between CPU and GPU using TBB resumable task and parallel_for. The TBB resumable task uses SYCL to implement calculations on GPU while the parallel_for algorithm does CPU part of calculations. This tbb-resumable-tasks-sycl sample code is implemented using C++ and SYCL language for CPU and GPU.
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta) 
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes

  
## Key implementation details 
TBB resumable tasks and DPC++ implementation explained. 

## License  
This code sample is licensed under MIT license

## How to Build

### On Linux
    * Build tbb-resumable-tasks-sycl program
      cd tbb-resumable-tasks-sycl &&
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
     * MSBuild tbb-resumable-tasks-sycl.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE
     * Open Visual Studio 2017
     * Select Menu "File > Open > Project/Solution", find "tbb-resumable-tasks-sycl" folder and select "tbb-resumable-tasks-sycl.sln"
     * Select Menu "Project > Build" to build the selected configuration
     * Select Menu "Debug > Start Without Debugging" to run the program
