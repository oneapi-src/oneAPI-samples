# `printf` Sample

This sample is an FPGA tutorial that explains and shows how to use `sycl::ext::oneapi::experimental::printf` to print in a SYCL*-compliant FPGA program.

| Area                 | Description
|:--                   |:--
| What you will learn  | How to declare and use `printf()` in code
| Time to complete     | 10 minutes
| Category             | Concepts and Functionality

## Purpose

This tutorial shows how to use some simple macros to enable easy use of the SYCL `printf()` function. The function allows printing from within code running on the FPGA.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning**: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples.
It is categorized as a Tier 2 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.


## Key Implementation Details

The sample illustrates the following important concepts.

- Using the experimental `printf()` in code.
- Showing the advantages of `printf()`:
  - easy to use
  - smaller area usage and better performance
- Discussing the [limitations](#known-issues-and-limitations) of the `printf()`.

### Motivation

Previously, we've provided examples for how to print data using the Stream class in the [*FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/kernel-controls/pipes-extension/i-o-pipes.html).

Compare to the Stream class, `printf()` has the following advantages:

- `printf()` is a function that is globally available; Stream is an object that can only be obtained by passing it as a kernel argument from the host. In order to use Stream somewhere within your application, you have to pass the Stream object through the whole call stack. For debugging in large applications, using `printf()` has a great advantage of allowing you to add some print statements that are easy to remove later without changing your main code.

- On the FPGA device, `printf()` has smaller area usage (fewer LSUs) and better performance (Stream could introduce massive II inner loops).

### Simple Code Example

Using `printf()` can be verbose in SYCL kernels. To simplify, add the following macro to your code.

```cpp
#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif

#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

PRINTF("Hello, World!\n");
PRINTF("Hello: %d\n", 123);
```

### Known Issues and Limitations

There are some known issues with the `experimental::printf()` and that's why the function is in the experimental namespace. The following limitations exist when using `experimental::printf()` on FPGA simulation and hardware:

- If you have multiple `printf` statements in the kernel, the order of printed data in the stdout might not obey the sequential order of those statements in the code.
- Buffer is only flushed to stdout after the kernel finishes in hardware.
- Printing `long` integers in Windows results is not supported yet. Printing `long` integers in Linux works as intended.

## Build the `printf` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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

### On Linux*

1. Change to the sample directory.
2. Build the program for Intel® Agilex® 7 device family, which is the default.
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

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulates an FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the HTML optimization reports. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      make report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile and run on FPGA hardware (longer compile time, targets an FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Intel® Agilex® 7 device family, which is the default.
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

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      nmake report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced problem size).
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Read the Reports

Locate `report.html` in the `printf_report.prj/reports/` directory.

From the report, you can find the compilation information of the design and the estimated FPGA resource usage. For example, navigate to the Area Analysis section of the report (**Kernel System > BasicKernel > Computation**), and you can see the estimated resource usage (ALUTs, FFs, RAMs, etc.) for each `printf()` call.

## Run the `printf` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./printf.fpga_emu
   ```
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./printf.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./printf.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   printf.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   printf.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   printf.fpga.exe
   ```

## Example Output

```
Result1: Hello, World!
Result2: %
Result3: 123
Result4: 123
Result5: 1.00
Result6: print slash_n \n
Result7: Long: 650000
Result8: Preceding with blanks:       1977
Result9: Preceding with zeros: 0000001977
Result10: Some different radices: 100 64 144 0x64 0144
Result11: ABCD
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).