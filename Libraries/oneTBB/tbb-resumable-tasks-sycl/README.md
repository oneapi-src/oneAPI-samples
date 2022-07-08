# `tbb-resumable-tasks-sycl` Sample

This sample illustrates how the computational kernel can be split for execution between CPU and GPU using TBB resumable task and parallel_for. The TBB resumable task uses SYCL to implement GPU calculations, while the parallel_for algorithm does the CPU part of calculations. This tbb-resumable-tasks-sycl sample code is implemented using C++ and SYCL language for CPU and GPU.
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | How to offload the computation to GPU using oneAPI DPC++ Compiler
| Time to complete                  | 15 minutes

## Purpose
The purpose of this sample is to show how during execution, a computational kernel can be split between CPU and GPU using TBB resumable tasks and TBB parallel_for.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

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

## Key implementation details
TBB resumable tasks and DPC++ implementation explained.

## Building the tbb-resumable-tasks-sycl program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On a Linux System
    * Build tbb-resumable-tasks-sycl program
      cd tbb-resumable-tasks-sycl &&
      mkdir build &&
      cd build &&
      cmake .. && make VERBOSE=1

    * Run the program
      make run

    * Clean the program
      make clean

### On a Windows System

#### Command line using MSBuild
    * MSBuild tbb-resumable-tasks-sycl.sln /t:Rebuild /      p:Configuration="debug"

#### Visual Studio IDE
    * Open Visual Studio 2017
    * Select Menu "File > Open > Project/Solution", find "tbb-resumable-tasks-sycl" folder and select "tbb-resumable-tasks-sycl.sln"
    * Select Menu "Project > Build" to build the selected configuration
    * Select Menu "Debug > Start Without Debugging" to run the program



## Running the Sample

### Application Parameters
None

### Example of Output

    start index for GPU = 0; end index for GPU = 8
    start index for CPU = 8; end index for CPU = 16
    Heterogenous triad correct.
    c_array: 0 1.5 3 4.5 6 7.5 9 10.5 12 13.5 15 16.5 18 19.5 21 22.5
    c_gold : 0 1.5 3 4.5 6 7.5 9 10.5 12 13.5 15 16.5 18 19.5 21 22.5
    Built target run

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)