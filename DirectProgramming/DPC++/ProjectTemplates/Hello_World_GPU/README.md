# `Visual Studio based GPU Project` Sample
A minimal project template for GPU using for Visual Studio.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Windows 10
| Hardware                          | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software                          | Intel(R) oneAPI DPC++ Compiler (beta)
| What you will learn               | Get started with DPC++ for GPU projects
| Time to complete                  | n/a

## Purpose
This project is a template designed to help you create your own Data Parallel C++ application for GPU targets. The template assumes the use of make to build your application. Review the main.cpp source file for help with the header files you should include and how to implement "device selector" code for targeting your application's runtime device

If GPU is not available on your system, you can fallback to cpu or default device.

## Key Implementation Details
The basic DPC++ project template for FPGA targets.

## License
This code sample is licensed under the MIT license.

## Building the `Visual Studio based GPU` Program

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Windows* System Using Visual Studio* Version 2017 or Newer
* Build the program using VS2017 or VS2019
  Right click on the solution file and open using either VS2017 or VS2019 IDE.
  Right click on the project in Solution explorer and select Rebuild.
  From top menu select Debug -> Start without Debugging.

* Build the program using MSBuild
  Open "x64 Native Tools Command Prompt for VS2017" or
  "x64 Native Tools Command Prompt for VS2019"
  Run - MSBuild Hello_World_GPU.sln /t:Rebuild /p:Configuration="Release"