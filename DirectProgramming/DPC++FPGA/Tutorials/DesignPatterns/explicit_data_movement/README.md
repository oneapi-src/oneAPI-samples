# `Explicit Data Movement` Sample

An FPGA tutorial demonstrating an alternative coding style, SYCL* Unified Shared Memory (USM) device allocations, in which data movement between host and device is controlled explicitly by the code author.

| Area                 | Description
|:---                  |:---
| What you will learn  | How to manage the movement of data explicitly for FPGA devices
| Time to complete     | 15 minutes
| Category             | Code Optimization

## Purpose

The purpose of this tutorial is to demonstrate an alternative coding style that allows you to control the movement of data explicitly in your SYCL-compliant program by using SYCL USM device allocations.

## Prerequisites

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware           | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> FPGA third-party/custom platforms with oneAPI support
| Software           | Intel® oneAPI DPC++/C++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> Questa*-Intel® FPGA Edition
> Questa*-Intel® FPGA Starter Edition
> ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

>**Note**: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*.

### Additional Documentation

- *[Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html)* helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- *[FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)* helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)* helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

## Key Implementation Details

### Implicit and Explicit Data Movement

A typical SYCL design copies input data from the Host Memory to the FPGA device memory. To do this, the data is sent to the FPGA board over PCIe. Once all the data is copied to the FPGA Device Memory, the FPGA kernel is run and produces output that is also stored in Device Memory. Finally, the output data is transferred from the FPGA Device Memory back to the CPU Host Memory over PCIe. This model is shown in the figure below.

```
|-------------|                               |---------------|
|             |   |-------| PCIe |--------|   |               |
| Host Memory |<=>|  CPU  |<====>|  FPGA  |<=>| Device Memory |
|             |   |-------|      |--------|   |               |
|-------------|                               |---------------|
```

SYCL provides two methods for managing the movement of data between host and device:

- **Implicit data movement**: Copying data to and from the FPGA Device Memory is not controlled by the code author but rather by a parallel runtime. The parallel runtime is responsible to ensure data is correctly transferred before being used and that kernels accessing the same data are synchronized appropriately.
- **Explicit data movement**: Copying data to and from the FPGA Device Memory is explicitly handled by the code author in the source code. The code author is responsible to ensure data is correctly transferred before being used and that kernels accessing the same data are synchronized appropriately.

Most of the other code sample tutorials in the [FPGA Tutorials](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming) in the one-API-samples GitHub repository use implicit data movement, which is achieved through the `buffer` and `accessor` SYCL constructs. This tutorial demonstrates an alternative coding style using explicit data movement, which is achieved through USM device allocations.

### SYCL USM Device Allocations

SYCL USM device allocations enable the use of raw *C-style* pointers in SYCL, called *device pointers*. Allocating space in the FPGA Device Memory for device pointers is done explicitly using the `malloc_device` function. Copying data to or from a device pointer must be done explicitly by the programmer, typically using the SYCL `memcpy` function. Additionally, synchronization between kernels accessing the same device pointers must be expressed explicitly by the programmer using either the `wait` function of a SYCL `event` or the `depends_on` signal between events.

In this tutorial, the `SubmitImplicitKernel` function demonstrates the basic SYCL buffer model, while the `SubmitExplicitKernel` function demonstrates how you may use these USM device allocations to perform the same task and manually manage the movement of data.

### Deciding Which Style to Use

The main advantage of explicit data movement is that it gives you full control over when data is transferred between different memories. In some cases, we can improve our design's overall performance by overlapping data movement and computation, which can be done with more fine-grained control using USM device allocations. The main disadvantage of explicit data movement is that specifying all data movements can be tedious and error prone.

Choosing a data movement strategy largely depends on the specific application and personal preference. However, when starting a new design, it is typically easier to begin by using implicit data movement since all of the data movement is handled for you, allowing you to focus on expressing and validating the computation. Once the design is validated, you can profile your application. If you determine that there is still performance on the table, it may be worthwhile switching to explicit data movement.

Alternatively, there is a hybrid approach that uses some implicit data movement and some explicit data movement. This technique, demonstrated in the **Double Buffering** (double_buffering) and  **N-Way Buffering** (n_way_buffering) tutorials, uses implicit data movement for some buffers where the control does not affect performance, and explicit data movement for buffers whose movement has a substantial effect on performance. In this hybrid approach, we do **not** use device allocations but rather specific `buffer` API calls (e.g., `update_host`) to trigger the movement of data.


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Explicit Data Movement` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the *[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```
   For **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   For a custom FPGA platform, ensure that the board support package is installed on your system then enter a command similar to the following:

   ```
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
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
      The report resides at `explicit_data_movement.prj/reports/report.html`. Note that because the optimization occurs at the *runtime* level, the FPGA compiler report will not show a difference between the optimized and unoptimized cases.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```
   (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/explicit_data_movement.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/explicit_data_movement.fpga.tar.gz).


### On Windows*

>**Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   To compile for the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   For a custom FPGA platform, ensure that the board support package is installed on your system then enter a command similar to the following:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `explicit_data_movement.prj.a/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your `build` directory in a shorter path, for example `C:\samples\build`. You can then build the sample in the new location, but you must specify the full path to the build files.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Run the `Explicit Data Movement` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./explicit_data_movement.fpga_emu
   ```
2. Run the sample on the FPGA device.
   ```
   ./explicit_data_movement.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   explicit_data_movement.fpga_emu.exe
   ```
2. Run the sample on the FPGA device.
   ```
   explicit_data_movement.fpga.exe
   ```

### Build and Run the Samples on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.


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

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **12 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.


## Example Output

### Output Example for FPGA Emulator

```
Running the ImplicitKernel with size=10000
Running the ExplicitKernel with size=10000
PASSED
```

### Output Example for FPGA Device

```
Running the ImplicitKernel with size=100000000
Running the ExplicitKernel with size=100000000
Average latency for the ImplicitKernel: 530.762 ms
Average latency for the ExplicitKernel: 470.453 ms
PASSED
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
