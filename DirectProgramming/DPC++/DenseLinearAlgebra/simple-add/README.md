# `Simple Add` Sample

The `Simple Add` sample demonstrates the simplest programming methods for using SYCL*-compliant buffers and Unified Shared Memory (USM).

For comprehensive instructions, see the [Intel&reg; oneAPI Programming
Guide](https://software.intel.com/en-us/oneapi-programming-guide) and search
based on relevant terms noted in the comments.

| Property            | Description 
|:---                 |:---
| What you will learn | How to use SYCL*-compliant extensions to offload computations using both buffers and USM.
| Time to complete    | 15 minutes

## Purpose

The `Simple Add` sample is a simple program that adds two large vectors of
integers and verifies the results. In this sample, you will see how to use 
the most basic code in C++ language that offloads computations to a GPU, which includes using USM and buffers.

The basic SYCL* implementations explained in the sample includes device selector,
USM, buffer, accessor, kernel, and command groups.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10
| Hardware                          | Skylake with GEN9 or newer <br>Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler

### Known Issues

As of the oneAPI 2021.4 release, the argument for accessors was changed from `noinit` to
`no_init`. The change was derived from a change between the SYCL 2020 provisional spec and that of the 2020 Rev3 spec.

If you run the sample program and it fails, try one of the following actions:
- Update the Intel® oneAPI Base Toolkit to the latest version.
- Change the `no_init` argument  to `noinit`

## Key Implementation Details

This sample provides examples of both buffers and USM implementations for simple side-by-side comparison.

- USM requires an explicit wait for the asynchronous kernel's
computation to complete.

- Buffers, at the time they go out of scope, bring main
memory in sync with device memory implicitly. The explicit wait on the event is
not required as a result.

The program attempts first to run on an available GPU, and it will fall back to the system CPU if it does not detect a compatible GPU. If the program runs successfully, the name of the offload device and a success message is displayed.

> **Note**: Successfully building and running the sample proves your development environment is configured correctly.

## Setting Environment Variables
For working with the Command-Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the commands similar to the following: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

## Include Files
Many oneAPI code samples, like this one, share common include files. The include files are installed locally and are located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system. You will need the `dpc_common.hpp` from that location to build these samples.

## Build the `Simple Add` Program
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment
   Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment,
see the [Using Visual Studio Code with Intel&reg; oneAPI
Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors and other issues. See the [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

### Build the `Simple Add` Sample for CPU and GPU
#### Linux*
1. Build the program using `make` (by default USM).
    ```
    make all
    ```
2. Build with buffers (optional).
   ```
   make build_buffers
   ```

#### On Windows* Using a Command Line Interface**
1. Select **Programs** > **Intel oneAPI 2021** > **Intel oneAPI Command Prompt**
   to launch a command window.
2. Build the program using the following `nmake` commands (Windows supports USM
   only):
   ```
   nmake -f Makefile.win
   ```
#### On Windows* Using Visual Studio*
Build the program using VS2017 or later.
1. Right-click on the solution file and open using the IDE.
2. Right-click on the project in **Solution Explorer** and select **Rebuild**.
3. From the top menu, select **Debug** > **Start without Debugging**.

Optionally, build the program using MSBuild.
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio version.
2. Run the following command: `MSBuild simple-add.sln /t:Rebuild /p:Configuration="Release"`

## Build the `Simple Add` Sample for FPGA

### On Linux*
1. Based on your requirements, build for FPGA emulation using `make`.
    ```
    make fpga_emu -f Makefile.fpga
    ```
2. If you have the hardware, build for FPGA hardware using 'make'. 
   (The hardware compilation might take a long time.)
    ```
    make hw -f Makefile.fpga
    ````
3. Generate static optimization reports for design analysis.
    ```
    make report -f Makefile.fpga
    ```
   The path to the generated report is similar to `simple-add_report.prj/reports/report.html`.

### On Windows* Using a Command Line Interface
>**Note**: On Windows, you can only compile and run on the FPGA emulator.
Generating an HTML optimization report and compiling and running on the FPGA
hardware is not currently supported.

1. Select **Programs** > **Intel oneAPI 2021** > **Intel oneAPI Command Prompt**
   to launch a command window.

2. Build the program using the following `nmake` commands:
   ```
   nmake -f Makefile.win.fpga
   ```
## Run the `Simple Add` Sample
### Linux*
1. Run the program using:
    ```
    make run
    ```

2. Optionally, run the program for buffers
   ```
   make run_buffers
   ```
### Linux* for FPGA
1. Run the program for FPGA emulation using `make`.
    ```
    make run_emu -f Makefile.fpga
    ```
2. If you have the appropriate hardware and you have built the program, run the program for FPGA hardware.
    ```
    make run_hw -f Makefile.fpga
    ```

### Windows*
3. Run the program using 'nmake`.
   ```
   nmake -f Makefile.win.fpga run
   ```
### Application Parameters
The sample has no configurable parameters.

### Run the `Simple Add` Sample in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node
(CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more
information, see the Intel&reg; oneAPI Base Toolkit [Get Started
Guide](https://devcloud.intel.com/oneapi/get_started/).

### Example Output
```
simple-add output snippet changed to:
Running on device:        Intel(R) Gen9 HD Graphics NEO
Array size: 10000
[0]: 0 + 100000 = 100000
[1]: 1 + 100000 = 100001
[2]: 2 + 100000 = 100002
...
[9999]: 9999 + 100000 = 109999
Successfully completed on device.
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).