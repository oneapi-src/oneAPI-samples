# `Base: Vector Add` Sample

The `Base: Vector Add` is the equivalent of a **Hello, World!** sample for data parallel programs, so the sample code demonstrates some core features of SYCL*. Additionally, building and running this sample verifies that your development environment is configured correctly for [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html).

| Area                     | Description
|:---                      |:---
| What you will learn      | How to begin using SYCL* to offload computations to CPUs and accelerators
| Time to complete         | ~15 minutes
| Category                 | Getting Started

## Purpose

The `Base: Vector Add` is a relatively simple program that adds two large vectors of integers and verifies the results. This program uses C++ and SYCL* for Intel® CPU and accelerators. By reviewing the code in this sample, you can learn how to use C++ code to offload computations to a GPU. This includes using Unified Shared Memory (USM) and buffers.

- USM requires an explicit wait for the asynchronous computation on the kernel to complete.

- When they go out of scope, buffers synchronize main memory with device memory implicitly; an explicit wait on the event is not required.

This sample provides example implementations of both Unified Shared Memory (USM) and buffers for side-by-side comparison.

>**Note**: See the `Simple Add` sample to examine another getting started sample you can use to learn more about using the Intel® oneAPI Toolkits to develop SYCL-compliant applications for CPU, GPU, and FPGA devices.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                          | GEN9 or newer <br> Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for CPU, GPU, FPGA emulation, generating FPGA reports and generating RTL for FPGAs, there are extra software requirements for the FPGA simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Key Implementation Details

The basic SYCL implementation explained in the code includes device selector, USM, buffer, accessor, kernel, and command groups.

The code attempts to execute on an available GPU and fallback to the system CPU if a compatible GPU is not detected. If successful, the name of the offload device and a success message is displayed, which indicates your development environment is set up correctly.

In addition, you can target an FPGA device using the instructions provided below. If you do not have FPGA hardware, the sample will run in emulation mode, which includes static optimization reports for design analysis.

> **Note**: For comprehensive information about oneAPI programming, see the [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Base: Vector Add` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Using Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
3. Open a terminal in VS Code (**Terminal > New Terminal**).
4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

#### Configure the build system

1. Change to the sample directory.
2.
   Configure the project to use the buffer-based implementation.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   or

   Configure the project to use the Unified Shared Memory (USM) based implementation.
   ```
   mkdir build
   cd build
   cmake .. -DUSM=1
   ```

   > **Note**: When building for FPGAs, the default FPGA family will be used (Intel® Agilex® 7).
   > You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   > Here are a few examples of FPGA board variant and BSP (this list is not exhaustive):
   > 
   > For Intel® PAC with Intel Arria® 10 GX FPGA, the USM is not supported, you can use below BSP:
   > 
   >     intel_a10gx_pac:pac_a10
   >
   > For Intel® FPGA PAC D5005, use one of the following BSP based on the USM support:
   >
   >     intel_s10sx_pac:pac_s10
   >     intel_s10sx_pac:pac_s10_usm
   > 
   > You will only be able to run an executable on the FPGA if you specified a BSP.

#### Build for CPU and GPU

1. Build the program.
   ```
   make cpu-gpu
   ```
2. Clean the program. (Optional)
   ```
   make clean
   ```

#### Build for FPGA

1. Compile for FPGA emulation.
   ```
   make fpga_emu
   ```
2. Compile for simulation (fast compile time, targets simulator FPGA device):
   ```
   make fpga_sim
   ```
3. Generate HTML performance reports.
   ```
   make report
   ```
   The reports reside at `simple-add_report.prj/reports/report.html`.

4. Compile the program for FPGA hardware. (Compiling for hardware can take a long
time.)
   ```
   make fpga
   ```

5. Clean the program. (Optional)
   ```
   make clean
   ```

### On Windows*

#### Configure the build system

1. Change to the sample directory.
2.
   Configure the project to use the buffer-based implementation.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   or

   Configure the project to use the Unified Shared Memory (USM) based implementation.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" .. -DUSM=1
   ```

   > **Note**: When building for FPGAs, the default FPGA family will be used (Intel® Agilex® 7).
   > You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   > Here are a few examples of FPGA board variant and BSP (this list is not exhaustive):
   > 
   > For Intel® PAC with Intel Arria® 10 GX FPGA, the USM is not supported, you can use below BSP:
   > 
   >     intel_a10gx_pac:pac_a10
   >
   > For Intel® FPGA PAC D5005, use one of the following BSP based on the USM support:
   >
   >     intel_s10sx_pac:pac_s10
   >     intel_s10sx_pac:pac_s10_usm
   > 
   > You will only be able to run an executable on the FPGA if you specified a BSP.

#### Build for CPU and GPU

1. Build the program.
   ```
   nmake cpu-gpu
   ```
2. Clean the program. (Optional)
   ```
   nmake clean
   ```

#### Build for FPGA

>**Note**: Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Compile for FPGA emulation.
   ```
   nmake fpga_emu
   ```
2. Compile for simulation (fast compile time, targets simulator FPGA device):
   ```
   nmake fpga_sim
   ```
3. Generate HTML performance reports.
   ```
   nmake report
   ```
The reports reside at `simple-add_report.prj/reports/report.html`.

4. Compile the program for FPGA hardware. (Compiling for hardware can take a long
time.)
   ```
   nmake fpga
   ```

5. Clean the program. (Optional)
   ```
   nmake clean
   ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `Base: Vector Add` Program

### Configurable Parameters

The source files (`vector-add-buffers.cpp` and `vector-add-usm.cpp`) specify the default vector size of **10000**. You can change the vector size in one or both files if necessary.

### On Linux

#### Run for CPU and GPU

1. Change to the output directory.

2. Run the program for Unified Shared Memory (USM) and buffers.
    ```
    ./vector-add-buffers
    ./vector-add-usm
    ```
#### Run for FPGA

1.  Change to the output directory.

2.  Run for FPGA emulation.
    ```
    ./vector-add-buffers.fpga_emu
    ./vector-add-usm.fpga_emu
    ```
3. Run on FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./vector-add-buffers.fpga_sim
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./vector-add-usm.fpga_sim
   ```
4. Run on FPGA hardware (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
    ```
    ./vector-add-buffers.fpga
    ./vector-add-usm.fpga
    ```

### On Windows

#### Run for CPU and GPU

1. Change to the output directory.

2. Run the program for Unified Shared Memory (USM) and buffers.
    ```
    vector-add-usm.exe
    vector-add-buffers.exe
    ```

#### Run for FPGA

1.  Change to the output directory.

2.  Run for FPGA emulation.
    ```
    vector-add-buffers.fpga_emu.exe
    vector-add-usm.fpga_emu.exe
    ```
3. Run on FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   vector-add-buffers.fpga_sim.exe
   vector-add-usm.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
4. Run on FPGA hardware (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
    ```
    vector-add-buffers.fpga.exe
    vector-add-usm.fpga.exe
    ```

### Build and Run the `Base: Vector Add` Sample in Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.

Use the Linux instructions to build and run the program.

You can specify a GPU node using a single line script.

```
qsub  -I  -l nodes=1:gpu:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes           |Command Options
  |:---                      |:---
  |GPU	                    |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	                    |`qsub -l nodes=1:xeon:ppn=2 -d .`
  |FPGA Compile Time         |`qsub -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime (Arria 10)   |`qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d .`


>**Note**: For more information on how to specify compute nodes, read [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **24 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.


## Example Output
```
Running on device:        Intel(R) Gen(R) HD Graphics NEO
Vector size: 10000
[0]: 0 + 0 = 0
[1]: 1 + 1 = 2
[2]: 2 + 2 = 4
...
[9999]: 9999 + 9999 = 19998
Vector add successfully completed on device.
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
