# `CMake FPGA` Template
This code sample is minimal project template for FPGA using the `CMake` build system.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to use a CMake for FPGA projects
| Time to complete      | 10 minutes
| Category              |Getting Started

## Purpose

This project is a template designed to help you quickly create your own SYCL*-compliant application for FPGA targets.

The template assumes the use of `CMake` to build your application. The supplied `CMakeLists.txt` file contains the compiler options and libraries needed to compile an application for FPGA targets.

The `main.cpp` source file shows the header files- that you should include and the recommended "device selector" code for targeting the application runtime device.

This is a project template only. You should review the FPGA `GettingStarted` code sample  [compile_flow](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/Tutorials/GettingStarted/fpga_compile), which explains the basic workflow, compile targets, and compiler options.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* LTS 18.04 or newer
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler <br> Intel® FPGA Add-on for oneAPI Base Toolkit

## Key Implementation Details

This sample is a basic SYCL*-compliant project template for FPGA targets.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `CMake FPGA` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Include Files

The include folder is at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Change to the sample directory.
2. Build the program.

   ```
   mkdir build
   cd build
   cmake ..
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `fpga_report.prj/reports/report.html`.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the  `CMake FPGA` Program

1. Change to the output directory.
2. Run the FPGA emulation.
     ```
     make fpga_emu
     ```

### Build and Run the Samples on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

Use the Linux instructions to build and run the program.

You can specify an FPGA runtime node using a single line script similar to the following example.

```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:fpga_runtime:ppn=2` (lower case L) assigns one full node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes           |Command Options
  |:---                      |:---
  |FPGA Compile Time         |`qsub -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime (Arria 10)   |`qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d .`
  |FPGA Runtime (Stratix 10) |`qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d .`
  |GPU	                     |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	                     |`qsub -l nodes=1:xeon:ppn=2 -d .`



>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **24 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
