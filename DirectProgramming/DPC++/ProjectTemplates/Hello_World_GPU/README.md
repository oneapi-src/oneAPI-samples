# `Hello World GPU` Sample
This sample is a minimal project template using Microsoft Visual Studio* for GPU projects.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to use Microsoft Visual Studio* for GPU projects
| Time to complete      | ~10 minutes

## Purpose

This project is a template designed to help you create your own SYCL*-compliant application for GPU targets. Review the main.cpp source file for help with the header files that you should include and how to implement "device selector" code for targeting the application runtime device.

If a GPU is not available on your system, fall back to the CPU or default device.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Windows* 10
| Hardware              | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software              | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

A basic SYCL*-compliant project template for GPU targets.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Hello World GPU` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files

The include folder is at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### On Windows*

**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. Right-click on the project in **Solution Explorer** and select **Rebuild**.
4. From the top menu, select **Debug** > **Start without Debugging**. (This runs the program too.)

**Using MSBuild**

1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild Hello_World_GPU.sln /t:Rebuild /p:Configuration="Release"`

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).