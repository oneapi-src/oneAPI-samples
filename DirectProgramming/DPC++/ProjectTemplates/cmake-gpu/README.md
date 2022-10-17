# `CMake GPU` Template

This sample is a minimal project template for GPU using the `CMake` build system.

| Area                     | Description
|:---                      |:---
| What you will learn      | Understand the compile flow for GPU projects using `CMake`
| Time to complete         | 10 minutes
| Category                 | Getting Started

## Purpose

This project is a template designed to help you create your own SYCL*-compliant application for GPU targets.

The template assumes the use of CMake to build your application. See the supplied `CMakeLists.txt` file for hints regarding the compiler options and libraries needed to compile an application for GPU targets.

Review the `main.cpp` source file for help with the header files that you should include and how to implement "device selector" code for targeting the application runtime device.

## Prerequisites

| Optimized for            | Description
|:---                      |:---
| OS                       | Ubuntu* LTS 18.04 or newer <br> RHEL 8.x
| Hardware                 | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software                 | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

A basic SYCL*-compliant project template for GPU targets.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `CMake GPU` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

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
   make
   ```
3. Clean the program. (Optional)
   ```
   make clean
   ```


#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `CMake GPU` Program

1. Run the compiled program.
   ```
   make run
   ```

### Build and Run the `CMake GPU` Sample in Intel® DevCloud (Optional)

<This is the short version. Use ONLY the short version OR the long version NOT both.>

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. 

Use the Linux instructions to build and run the program.

You can specify a GPU node using a single line script.

```
qsub  -I  -l nodes=1:gpu:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
- `-d .` makes the current folder as the working directory for the task.

<Remove the node examples that do not apply to the sample.>

|Available Nodes    |Command Options
|:---               |:---
|GPU	              |`qsub -l nodes=1:gpu:ppn=2 -d .`
|CPU	              |`qsub -l nodes=1:xeon:ppn=2 -d .`
|FPGA Compile Time  |`qsub -l nodes=1:fpga_compile:ppn=2 -d .`
|FPGA Runtime       |`qsub -l nodes=1:fpga_runtime:ppn=2 -d .`


>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

## Example Output

None. This is a template.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).