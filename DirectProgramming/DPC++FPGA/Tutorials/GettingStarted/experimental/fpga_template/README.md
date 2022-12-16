# FPGA Template

This project serves as a template for Intel® oneAPI FPGA designs. 

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support (and SYCL USM support) Note: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | Best practices for creating and managing a oneAPI FPGA project
| Time to complete                  | 10 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> To use the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> To use the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Note**: In oneAPI full systems, kernels that use SYCL Unified Shared Memory (USM) host allocations or USM shared allocations (and therefore the code in this tutorial) are only supported by Board Support Packages (BSPs) with USM support (e.g. the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) `intel_s10sx_pac:pac_s10_usm`). Kernels that use these types of allocations can always be used to generate standalone IPs.

## Purpose

Use this project as a starting point when you build designs for the Intel® oneAPI FPGA compiler. It includes a CMake build system to automate selecting the various command-line flags for the oneAPI FPGA compiler, and a simple single-source design to serve as an example. You can customize the build flags by modifying the top part of `src/CMakeLists.txt`: if you want to pass additional flags to the Intel® oneAPI FPGA compiler, you can change the `USER_FLAGS` and `USER_HARDWARE_FLAGS` variables defined in `src/CMakeLists.txt`. Similarly, you can add additional include paths to the `USER_INCLUDE_PATHS` variable. You can also explictly define these variables at the command-line if you don't want to make change to the CMake build system.

> **Note**: The code sample in this design only uses USM for improved code simplicity as compared with buffers/accessors. The included CMake build system can also be used for designs that do not use USM.

| Variable              | Description
|:---                   |:---
| `USER_HARDWARE_FLAGS` | This space-separated list of flags applies only to flows that generate FPGA hardware (i.e. report, simulation, hardware). You can specify flags such as `-Xsclock` or `-Xshyper-optimized-handshaking=off`
| `USER_FLAGS`          | This space-separated list of flags applies to all flows, including emulation. You can specify flags such as `-v` or define macros such as `-DYOUR_OWN_MACRO=3`
| `USER_INCLUDE_PATHS`  | This space-separated list of include paths applies  to all flows, including emulation. Specify include paths relative to the `src/CMakeLists.txt` file, or using absolute paths in the filesystem.

```bash
###############################################################################
### Customize these build variables
###############################################################################
set(SOURCE_FILE fpga_template.cpp)
set(TARGET_NAME fpga_template)

# Use cmake -DFPGA_DEVICE=<board-support-package>:<board-variant> to choose a
# different device. Here are a few device examples (this list is not
# exhaustive):
#   intel_s10sx_pac:pac_s10
#   intel_s10sx_pac:pac_s10_usm
#   intel_a10gx_pac:pac_a10
# Note that depending on your installation, you may need to specify the full 
# path to the board support package (BSP), this usually is in your install 
# folder.
#
# You can also specify a device family (E.g. "Arria10" or "Stratix10") or a
# specific part number (E.g. "10AS066N3F40E2SG") to generate a standalone IP.
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_s10sx_pac:pac_s10_usm")
endif()

# Use cmake -DUSER_FPGA_FLAGS=<flags> to set extra flags for FPGA backend
# compilation. 
set(USER_FPGA_FLAGS "${USER_FPGA_FLAGS}")

# Use cmake -DUSER_FLAGS=<flags> to set extra flags for general compilation.
set(USER_FLAGS "${USER_FLAGS}")

# Use cmake -DUSER_INCLUDE_PATHS=<paths> to set extra paths for general
# compilation.
set(USER_INCLUDE_PATHS "${USER_INCLUDE_PATHS}")
```

Everything below this in the `src/CMakeLists.txt` is necessary for selecting the compiler flags that are necessary to support the build targets specified below, and should not need to be modified.

## Building the `fpga_template` Tutorial

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `/opt/intel/oneapi/setvars.sh`
> - For private installations: `~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
>For more information on environment variables, see **Use the setvars Script** for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Running Samples in Intel® DevCloud
If running a sample in the Intel® DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).
When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.
### Using Visual Studio Code*  (Optional)
You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.
The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.
To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

Use these commands to run the design, depending on your OS.

### On a Linux* System 
This design uses CMake to generate a build script for GNU/make.

1. Generate the `Makefile` by running `cmake`.

   ```bash
   mkdir build
   cd build
   ```

   To compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```bash
   cmake ..
   ```

   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:

   ```
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

   > **NOTE**: This design will **not** work on the Intel® PAC with Intel Arria® 10 GX FPGA, because the design depends on USM.

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target          | Expected Time  | Output                                                                       | Description
   |:---             |:---            |:---                                                                          |:---
   | `make fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `make report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.
   | `make fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `make fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and generate an FPGA image that you can run on a supported accelerator board.

   The `fpga_emu`, `fpga_sim` and `fpga` targets produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu`, `TARGET_NAME.fpga_sim`, and `TARGET_NAME.fpga`, where `TARGET_NAME` is the value you specify in `src/CMakeLists.txt`.

### On a Windows* System
This design uses CMake to generate a build script for  `nmake`.

1. Generate the `Makefile` by running `cmake`.

   ```bash
   mkdir build
   cd build
   ```

   To compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```bash
   cmake -G "NMake Makefiles" ..
   ```

   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

   > **NOTE**: This design will **not** work on the Intel® PAC with Intel Arria® 10 GX FPGA, because the design depends on USM.


2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target           | Expected Time  | Output                                                                       | Description
   |:---              |:---            |:---                                                                          |:---
   | `nmake fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `nmake report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.
   | `nmake fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `nmake fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and generate an FPGA image that you can run on a supported accelerator board.

   The `fpga_emu`, `fpga_sim`, and `fpga` targets also produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu.exe`, `TARGET_NAME.fpga_sim.exe`, and `TARGET_NAME.fpga.exe`, where `TARGET_NAME` is the value you specify in `src/CMakeLists.txt`.

   > **Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

   > **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Additional Documentation
- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

### Troubleshooting
If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument (on Windows use `nmake` instead):

```
make VERBOSE=1
```

```
nmake VERBOSE=1
```

For more comprehensive troubleshooting, use the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors. [Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):

   ```
   ./fpga_template.fpga_emu     (Linux)
   fpga_template.fpga_emu.exe   (Windows)
   ```

2. Run the sample on the FPGA simulator device:

   > **NOTE**: you need to define the `CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 environment variable in oneAPI 2023.1

   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./fpga_template.fpga_sim                       (Linux)
   cmd /V /C "set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1&& fpga_template.fpga_sim.exe"   (Windows)
   ```

3. Run the sample on the FPGA device:
   ```
   ./fpga_template.fpga         (Linux)
   fpga_template.fpga.exe       (Windows)
   ```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).