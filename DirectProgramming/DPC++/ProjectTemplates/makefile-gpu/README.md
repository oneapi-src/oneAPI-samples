# `Make based GPU Project` Template
A minimal project template for GPU using make build system.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu LTS 18.04, 19.10; RHEL 8.x
| Hardware                          | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | Get started with compile flow for GPU projects
| Time to complete                  | n/a

## Purpose
This project is a template designed to help you create your own Data Parallel C++ application for GPU targets. The template assumes the use of make to build your application. See the supplied `Makefile` file for hints regarding the compiler options and libraries needed to compile a Data Parallel C++ application for GPU targets. And review the `main.cpp` source file for help with the header files you should include and how to implement the "device selector" code for targeting your application's runtime device.

## Key Implementation Details
The basic DPC++ project template for FPGA targets.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `Make based GPU` Program

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)


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

### On a Linux* System
The following instructions assume you are in the root of the project folder.

To build the template using:
```
    make all or make build
```
To run the template using:
```
    make run
```
Clean the template using:
```
    make clean
```
