# matrix_mul_mkl sample
matrix_mul_mkl is a simple program that multiplies together two large matrices and verifies the results.
This program is implemented using C++ with oneAPI Math Kernel Library (oneMKL):

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10*
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler beta, oneMKL
| What you will learn               | Offloads computations on 2D arrays to GPU using oneMKL
| Time to complete                  | 15 minutes

## Key implementation details
oneMKL implementation explained.

## License
This code sample is licensed under MIT license.

## How to Build for oneMKL

### on Linux
   * Build the program using Make
    cd matrix_mul_mkl &&
    make build_mkl

   * Run the program
    make run_mkl

   * Clean the program
    make clean

### on Windows

#### Command Line using MSBuild
   * MSBuild matrix_mul.sln /t:Rebuild /p:Configuration="release"

#### Command Line using nmake
   Build matrix_mul_mkl oneMKL version
   * nmake -f Makefile.win build_mkl
   * nmake -f Makefile.win run_mkl

   Build matrix_mul_mkl DPCPP version
   * nmake -f Makefile.win build_dpcpp
   * nmake -f Makefile.win run_dpcpp

#### Visual Studio IDE
   * Open Visual Studio 2017
   * Select Menu "File > Open > Project/Solution", find "matrix_mul" folder and select "matrix_mul.sln"
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program

