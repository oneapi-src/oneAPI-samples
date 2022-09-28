# `Visual Studio based GPU Project` Sample
This sample is a minimal project template using Microsoft Visual Studio* for GPU projects.

For comprehensive instructions, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Property                     | Description
|:---                               |:---
| What you will learn               | How to use Microsoft Visual Studio* for GPU projects

## Purpose
This project is a template designed to help you create your own SYCL*-compliant application for GPU targets. The template assumes the use of make to build your application. Review the main.cpp source file for help with the header files you should include and how to implement "device selector" code for targeting the application runtime device.

If a GPU is not available on your system, fall back to the CPU or default device.

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Windows* 10
| Hardware                          | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software                          | Intel&reg; oneAPI DPC++ Compiler

## Key Implementation Details
A basic SYCL*-compliant project template for FPGA targets.

## Building the `Visual Studio based GPU Project` Sample

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).


### On a Windows* System Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild Hello_World_GPU.sln /t:Rebuild /p:Configuration="Release"`

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
