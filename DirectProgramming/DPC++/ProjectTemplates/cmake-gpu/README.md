# `CMake based GPU Project` Template
A minimal project template for GPU using the CMake build system.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu LTS 18.04, 19.10; RHEL 8.x
| Hardware                          | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | Get started with compile flow for GPU projects
| Time to complete                  | n/a

## Purpose
This project is a template designed to help you create your own Data Parallel C++ application for GPU targets. The template assumes the use of CMake to build your application. See the supplied `CMakeLists.txt` file for hints regarding the compiler options and libraries needed to compile a Data Parallel C++ application for GPU targets. And review the `main.cpp` source file for help with the header files you should include and how to implement "device selector" code for targeting your application's runtime device.

## Key Implementation Details
The basic DPC++ project template for GPU targets.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `CMake based GPU` Program

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
The following instructions assume you are in the root of the project folder.

```
    mkdir build
    cd build
    cmake ..
```
  To build the template using:
```
    make all or make build
```

  To run the template using:
```
    make run
```
