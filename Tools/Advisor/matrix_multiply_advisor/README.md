# `Matrix Multiply` Sample
A sample containing multiple implementations of matrix multiplication code sample and  is implemented using the DPC++ language for CPU and GPU. 
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler; Intel&reg; Advisor
| What you will learn               | How to profile an application using Intel&reg; Advisor
| Time to complete                  | 15 minutes

## Purpose

The Matrix Multiplication sample performs basic matrix multiplication. Three versions are provided that use different features of DPC++.

## Key Implementation details

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups. 
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## How to Build  

This sample contains 3 version of matrix multiplication using DPC++:

    multiply1 – basic implementation of matrix multiply using DPC++
    multiply1_1 – basic implementation that replaces the buffer store with a local accessor “acc” to reduce memory traffic
    multiply1_2 – the basic implementation, plus adding the local accessor and matrix tiling

Edit the line in multiply.h to select the version of the multiply function:
#define MULTIPLY multiply1


### On a Linux* System
	To build DPC++ version:
	cd <sample dir>
	cmake .
	make 

    Clean the program  
    make clean  

### On a Windows* System Using Visual Studio 2017 or newer
   * Open Visual Studio 2017
   * Select Menu "File > Open > Project/Solution", find "matrix_multiply" folder and select "matrix_multiply.sln"
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program

### on Windows - command line - Build the program using MSBuild
    DPCPP Configurations:
    Release - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Release"
    Debug - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Debug"



### Example of Output

   ./matrix.dpcpp 

   Using multiply kernel: multiply1

   Running on Intel(R) Gen9

   Elapsed Time: 0.539631s

## Running an Intel Advisor analysis
------------------------------------------

See the Advisor Cookbook here: https://software.intel.com/en-us/advisor-cookbook

