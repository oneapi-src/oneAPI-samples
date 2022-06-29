# `Unrolling Loops` Sample
This sample demonstrates a simple example of unrolling loops to improve throughput of a program using SYCL*-compliant code for GPU offload. The concepts demonstrated by the sample are:

- Understanding basics of loop unrolling.
- Unrolling loops in your program.
- Determining optimal unroll factor for your program.

For comprehensive instructions, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Property                     | Description
|:---                               |:---
| What you will learn   | How to perform loop unrolling using SYCL*
| Time to complete      | 30 minutes

## Purpose

The loop unrolling mechanism is used to increase program parallelism by duplicating the compute logic within a loop. The *unroll factor* refers to the number of times the loop logic is duplicated. Depending on whether the *unroll factor* is equal to the number of loop iterations or not, loop unroll methods can be categorized as *full-loop unrolling* and *partial-loop unrolling*. A full unroll is a special case where the *unroll factor* is equal to the number of loop iterations.

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS	                | Linux* Ubuntu* 18.04
| Hardware	            | Skylake with GEN9 or newer
| Software	            | Intel&reg; oneAPI DPC++/C++ Compiler

## Building the `loop_unroll` Tutorial

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
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel&reg; oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building the `loop-unroll` Program for CPU and GPU

### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

### On a Linux* System
1. Build the program using the following `cmake` commands.
   ```
   $ cd loop-unroll
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```
2. Run the program.
   ```
   $ make run
   ```
3. Clean the program.
   ```
   $ make clean
   ```
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
### Troubleshooting

If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### On Windows* Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019.
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild loop-unroll.sln.sln /t:Rebuild /p:Configuration="Release"`
## Running the Sample

### Example of Output
```
Input array size: 67108864
Running on device: Intel(R) Gen9
Unroll factor: 1 Kernel time: 13710.9 ms
Throughput for kernel with unroll factor 1: 0.005 GFlops
Unroll factor: 2 Kernel time: 8906.831 ms
Throughput for kernel with unroll factor 2: 0.008 GFlops
Unroll factor: 4 Kernel time: 4661.967 ms
Throughput for kernel with unroll factor 4: 0.014 GFlops
Unroll factor: 8 Kernel time: 2669.343 ms
Throughput for kernel with unroll factor 8: 0.025 GFlops
Unroll factor: 16 Kernel time: 2421.305 ms
Throughput for kernel with unroll factor 16: 0.028 GFlops
PASSED: The results are correct.
```
## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).