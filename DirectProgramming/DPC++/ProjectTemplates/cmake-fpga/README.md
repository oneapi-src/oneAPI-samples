# `CMake based FPGA` Project Template
A minimal project template for FPGA using CMake build system.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu LTS 18.04
| Hardware                          | Intel Programmable Acceleration Card with Intel Arria10 GX FPGA or Stratix(R) 10 SX FPGA
| Software                          | Intel(R) oneAPI DPC++ Compiler (beta), Intel(R) FPGA Add-on for oneAPI Base Toolkit
| What you will learn               | Get started with basic setup for FPGA projects
| Time to complete                  | n/a

## Purpose
This project is a template designed to help you quickly create your own Data Parallel C++ application for FPGA targets. The template assumes the use of CMake to build your application. See the supplied `CMakeLists.txt` file for hints regarding the compiler options and libraries needed to compile a Data Parallel C++ application for FPGA targets. Review the `main.cpp` source file for help with the header files you should include and how to implement "device selector" code for targeting your application's runtime device.

To see a simple FPGA kernel and an explanation of the recommended workflow with FPGA targets, consult the "compile flow" FPGA Tutorial.

## Key Implementation Details
The basic DPC++ project template for FPGA targets.

## License
This code sample is licensed under the MIT license.

## Building the `CMake based FPGA` Program

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
The following instructions assume you are in the project's root folder. Since the template does not define a kernel, all code in "main" is executed on the host regardless of the make target.

1. Create a `build` directory for `cmake` build artifacts:

    ```
    mkdir build
    cd build
    cmake ..
    ```
2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * To build the template for FPGA Emulator (fast compile time, targets a CPU-emulated FPGA device), and then run it:

    ```
    make fpga_emu
    ./cmake.fpga_emu
    ```

   * To generate the FPGA optimization report (fast compile time, partial FPGA HW compile):

    ```
    make report
    ```
    Locate the FPGA optimization report, `report.html`, in the `fpga_report.prj/reports/` directory.

   * To build and run the template for FPGA Hardware (takes about one hour, system must have at least 32 GB of physical dynamic memory):

    ```
    make fpga
    ./cmake.fpga
    ```