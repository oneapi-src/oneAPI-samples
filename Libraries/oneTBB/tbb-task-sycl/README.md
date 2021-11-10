# `TBB-Task-Sycl` Sample
This sample illustrates how two TBB tasks can execute similar computational kernels, with one task executing the SYCL code and the other task executing the TBB code. This `tbb-task-sycl` sample code is implemented using C++ and SYCL language for CPU and GPU.
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to offload the computation to GPU using oneAPI DPC++ Compiler
| Time to complete                  | 15 minutes

## Purpose
The purpose of this sample is to show how similar computational kernels can be executed by two TBB tasks, with one executing on SYCL code and another on TBB code.

## Key Implementation Details
The implementation based on TBB tasks and SYCL explained.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

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

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

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

