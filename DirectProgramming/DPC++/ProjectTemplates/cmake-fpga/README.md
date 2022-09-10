# `CMake based FPGA Project` Template
This code sample is minimal project template for FPGA using the `CMake` build system.

For comprehensive instructions, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Property                     | Description
|:---                               |:---
| What you will learn               | How to use a CMake for FPGA projects

## Purpose
This project is a template designed to help you quickly create your own SYCL*-compliant application for FPGA targets. The template assumes the use of `CMake` to build your application. The supplied `CMakeLists.txt` file contains the compiler options and libraries needed to compile an application for FPGA targets. The `main.cpp` source file shows the header files you should include and the recommended "device selector" code for targeting the application runtime device.

This is a project template only. You should review the FPGA `GettingStarted` code sample  [compile_flow](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/Tutorials/GettingStarted/fpga_compile), which explains the basic workflow, compile targets, and compiler options.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* LTS 18.04
| Hardware                          | Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA or Intel&reg; Stratix&reg; 10 FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br>Intel&reg; FPGA Add-on for oneAPI Base Toolkit

## Building the `CMake based FPGA` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On Linux*
The following instructions assume you are in the project's root folder.

1. Create a `build` directory for `cmake` build artifacts:
    ```
    mkdir build
    cd build
    cmake ..
    ```
2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   - Use the following commands to build the template for FPGA Emulator (fast compile time, targets a CPU-emulated FPGA device) and run it:
     ```
     make fpga_emu
     ./cmake.fpga_emu
     ```
   - Use the following command to generate the FPGA optimization report (fast compile time, partial FPGA HW compile):
     ```
     make report
     ```
     Locate the FPGA optimization report, `report.html`, in the `fpga_report.prj/reports/` directory.
   - Use the following commands to build and run the template for FPGA Hardware (takes about **one hour**, and the system must have at least **32 Gb** of physical dynamic memory):
     ```
     make fpga
     ./cmake.fpga
     ```
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Key Implementation Details
The basic SYCL*-compliant project template for FPGA targets.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).