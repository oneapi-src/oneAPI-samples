# `FPGA Template` Sample

This project serves as a template for Intel® oneAPI FPGA designs.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
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
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

> **Note**: In oneAPI full systems, kernels that use SYCL Unified Shared Memory (USM) host allocations or USM shared allocations (and therefore the code in this tutorial) are only supported by Board Support Packages (BSPs) with USM support (e.g. the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) `intel_s10sx_pac:pac_s10_usm`). Kernels that use these types of allocations can always be used to generate standalone IPs.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 1 sample that helps you getting started.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

## Purpose

Use this project as a starting point when you build designs for the Intel® oneAPI DPC++/C++ compiler when targeting FPGAs. It includes a CMake build system to automate selecting the various command-line flags for the oneAPI DPC++/C++ compiler, and a simple single-source design to serve as an example. You can customize the build flags by modifying the top part of `CMakeLists.txt`: if you want to pass additional flags to the Intel® oneAPI DPC++/C++ compiler, you can change the `USER_FLAGS` and `USER_HARDWARE_FLAGS` variables defined in `CMakeLists.txt`. Similarly, you can add additional include paths to the `USER_INCLUDE_PATHS` variable. You can also explicitly define these variables at the command-line if you don't want to make change to the CMake build system.

> **Note**: The code sample in this design only uses USM for improved code simplicity as compared with buffers/accessors. The included CMake build system can also be used for designs that do not use USM.

| Variable              | Description
|:---                   |:---
| `USER_HARDWARE_FLAGS` | This semicolon-separated list of flags applies only to flows that generate FPGA hardware (i.e. report, simulation, hardware). You can specify flags such as `-Xsclock` or `-Xshyper-optimized-handshaking=off`
| `USER_FLAGS`          | This semicolon-separated list of flags applies to all flows, including emulation. You can specify flags such as `-v` or define macros such as `-DYOUR_OWN_MACRO=3`
| `USER_INCLUDE_PATHS`  | This semicolon-separated list of include paths applies  to all flows, including emulation. Specify include paths relative to the `CMakeLists.txt` file, or using absolute paths in the filesystem.

```bash
###############################################################################
### Customize these build variables
###############################################################################
set(SOURCE_FILES src/fpga_template.cpp)
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
set(USER_FPGA_FLAGS ${USER_FPGA_FLAGS})

# Use cmake -DUSER_FLAGS=<flags> to set extra flags for general compilation.
set(USER_FLAGS ${USER_FLAGS})

# Use cmake -DUSER_INCLUDE_PATHS=<paths> to set extra paths for general
# compilation.
set(USER_INCLUDE_PATHS ../../../../include;${USER_INCLUDE_PATHS})
```

Everything below this in the `CMakeLists.txt` is necessary for selecting the compiler flags that are necessary to support the build targets specified below, and should not need to be modified.

## Building the `fpga_template` Tutorial

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables.
> Set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation every time you open a new terminal window.
> This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

Use these commands to run the design, depending on your OS.

### On a Linux* System
This design uses CMake to generate a build script for GNU/make.

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target          | Expected Time  | Output                                                                       | Description
   |:---             |:---            |:---                                                                          |:---
   | `make fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `make report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package. The generated RTL may be exported to Intel® Quartus Prime software.
   | `make fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `make fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime. If you specified a BSP with `FPGA_DEVICE`, this will generate an FPGA image that you can run on the corresponding accelerator board.

   The `fpga_emu`, `fpga_sim` and `fpga` targets produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu`, `TARGET_NAME.fpga_sim`, and `TARGET_NAME.fpga`, where `TARGET_NAME` is the value you specify in `CMakeLists.txt`.

### On a Windows* System
This design uses CMake to generate a build script for  `nmake`.

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target           | Expected Time  | Output                                                                       | Description
   |:---              |:---            |:---                                                                          |:---
   | `nmake fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `nmake report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package. The generated RTL may be exported to Intel® Quartus Prime software.
   | `nmake fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `nmake fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime. If you specified a BSP with `FPGA_DEVICE`, this will generate an FPGA image that you can run on the corresponding accelerator board.

   The `fpga_emu`, `fpga_sim`, and `fpga` targets also produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu.exe`, `TARGET_NAME.fpga_sim.exe`, and `TARGET_NAME.fpga.exe`, where `TARGET_NAME` is the value you specify in `CMakeLists.txt`.

   > **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `fpga_template` Executable

### On Linux
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./fpga_template.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./fpga_template.fpga_sim
   ```
3. Alternatively, run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./fpga_template.fpga
   ```
### On Windows
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   fpga_template.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   fpga_template.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Alternatively, run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   fpga_template.fpga.exe
   ```

## Example Output

```
Running on device: Intel(R) FPGA Emulation Device
add two vectors of size 256
PASSED
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
