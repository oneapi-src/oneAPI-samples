# matrix_mul sample
matrix_mul is a simple program that multiplies together two large matrices and
verifies the results.  This program is implemented using two ways: 
    1. Data Parallel C++ (DPC++)
    2. OpenMP (omp)

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10*
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler, Intel&reg; oneAPI C++ Compiler Classic
| What you will learn               | Offloads computations on 2D arrays to GPU using Intel DPC++ and OpenMP
| Time to complete                  | 15 minutes  

### Purpose
matrix_mul is a slightly more complex computation than vector_add by
multiplying two large matrices.  The code will attempt to run the calculation
on both the GPU and CPU, and then verifies the results. The size of the
computation can be adjusted for heavier workloads (defined below). If
successful, the name of the offload device and a success message are
displayed.

This sample uses buffers to manage memory.  For more information regarding
different memory management options, refer to the vector_add sample.  

matrix_mul includes C++ implementations of both Data Parallel (DPC++) and
OpenMP; each is contained in its own .cpp file. This provides a way to compare
existing offload techniques such as OpenMP with Data Parallel C++ within a
relatively simple sample. The default will build the DPC++ application.
Separate OpenMP build instructions are provided below. Note: matrix_mul does not
support OpenMP on Windows.

The code will attempt first to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected.  The device used for compilation is displayed in the output.

## Key implementation details
SYCL implementation explained.
OpenMP offload implementation explained.

## License  
This code sample is licensed under MIT license. 

## Building the `matrrix_mul` Program for DPC++ and OpenMP

## Include Files
The include folder is located at "%ONEAPI_ROOT%\dev-utilities\latest\include" on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/hpc-toolkit/)

### How to build for DPC++ on Linux  
   * Build the program using Make  
    cd matrix_mul &&  
    make all  

   * Run the program  
    make run  

   * Clean the program  
    make clean 

### How to Build for OpenMP on Linux  
   * Build the program using Make  
    cd matrix_mul &&  
    make build_omp  

   * Run the program  
    make run_omp  

   * Clean the program  
    make clean

### How to build for DPC++ on Windows
The OpenMP offload target is not supported on Windows yet.

#### Command Line using MSBuild
   * MSBuild matrix_mul.sln /t:Rebuild /p:Configuration="release"  

#### Command Line using nmake
   Build matrix_mul DPCPP version
   * nmake -f Makefile.win build_dpcpp
   * nmake -f Makefile.win run_dpcpp  

#### Visual Studio IDE
   * Open Visual Studio 2017     
   * Select Menu "File > Open > Project/Solution", find "matrix_mul" folder and select "matrix_mul.sln" 
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program

### How to build for OpenMP on Windows
The OpenMP offload target is not supported on Windows at this time.

## Running the Sample

### Application Parameters 
You can modify the size of the computation by adjusting the size parameter
(must be in multiples of 8) in the dpcpp and omp .cpp files. The configurable parameters include:
   size = m_size = 150*8; // Must be a multiple of 8.
   M = m_size / 8;
   N = m_size / 4;
   P = m_size / 2;

### Example of Output
#### DPC++
```
 ./matrix_mul_dpc
Running on device: Intel(R) Gen9 HD Graphics NEO
Problem size: c(150,600) = a(150,300) * b(300,600)
Result of matrix multiplication using DPC++: Success - The results are correct!
```

#### OpenMP
```
./matrix_mul_omp
Problem size: c(150,600) = a(150,300) * b(300,600)
Running on 1 device(s)
The default device id: 0
Result of matrix multiplication using OpenMP: Success - The results are correct!
Result of matrix multiplication using GPU offloading: Success - The results are correct!
```