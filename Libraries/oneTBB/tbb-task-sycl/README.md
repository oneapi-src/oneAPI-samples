# tbb-task-sycl sample
This sample illustrates how 2 TBB tasks can execute similar computational kernels with one task executing SYCL code and another one the TBB code. This tbb-task-sycl sample code is implemented using C++ and SYCL language for CPU and GPU.
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta) 
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes

  
## Key implementation details 
The implementation based on TBB tasks and SYCL explained. 

## License  
This code sample is licensed under MIT license   

## How to Build  

### On Linux 
    * Build tbb-task-sycl program 
      cd tbb-task-sycl &&
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
     * MSBuild tbb-task-sycl.sln /t:Rebuild /p:Configuration="debug"
   
#### Visual Studio IDE
     * Open Visual Studio 2017
     * Select Menu "File > Open > Project/Solution", find "tbb-task-sycl" folder and select "tbb-task-sycl.sln"
     * Select Menu "Project > Build" to build the selected configuration
     * Select Menu "Debug > Start Without Debugging" to run the program
