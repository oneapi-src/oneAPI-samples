# `scheduler_target_fmax_mhz` Sample

This sample is an FPGA tutorial that explains the `scheduler_target_fmax_mhz` attribute and its effect on the performance of Intel® FPGA kernels.

| Area                 | Description
|:---                  |:---
| What you will learn  |  The behavior of the `scheduler_target_fmax_mhz` attribute and when to use it. <br> The effect this attribute can have on kernel performance on FPGA.
| Time to complete     | 15 minutes
| Category             | Concepts and Functionality

## Purpose

This tutorial demonstrates how to use the `[[intel::scheduler_target_fmax_mhz(N)]]` attribute to set the fMAX target for a single kernel. The purpose this attribute serves is to direct the compiler to prioritize a high fMAX over a low initiation interval (II). The `[[intel::initiation_interval(N)]]` attribute can change the II of a loop to improve performance. If you are not familiar with the attribute, refer to the prerequisite tutorial "Loop initiation_interval attribute".

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
It is categorized as a Tier 3 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.


## Key Implementation Details

The sample illustrates the following important concepts.

- The behavior of the `scheduler_target_fmax_mhz` attribute and when to use it on your kernel
- The effect this attribute can have on your kernel's performance on FPGA

### Specifying Schedule fMAX Target for Kernels

The compiler provides two methods to specify fMAX target for kernels:

1. By using the `[[intel::scheduler_target_fmax_mhz(N)]]` source-level attribute on a given kernel. This is the focus of this tutorial.
2. By using the `-Xsclock=<clock target in Hz/KHz/MHz/GHz or s/ms/us/ns/ps>` option in the icpx command to direct the compiler to compile all kernels globally at a specific fMAX target.

If you use both the command-line option `-Xsclock` and the source-level attribute `[intel::scheduler_target_fmax_mhz(N)]]` the attribute takes priority.

### Use Cases of `intel::scheduler_target_fmax_mhz(N)`

A bottleneck in a loop is one or more loop-carried dependencies that cause the loop to have either an II greater than one, or a lower fMAX to achieve II of one. By default, the compiler optimizes for the best fMAX-II ratio.

You should use the `[[intel::scheduler_target_fmax_mhz(N)]]` attribute if you want to tweak your kernel's throughput and area utilization, as well as the tradeoff between fMAX and II. When you use the `[[intel::scheduler_target_fmax_mhz(N)]]` attribute (or the `-Xsclock` icpx command option) to specify a desired fMAX, the compiler schedules the design at that specified fMAX value with minimal achievable II. When you use the `[[intel::initiation_interval(N)]]` attribute (refer to the prerequisite tutorial "Loop initiation_interval attribute") to specify a desired II, the compiler lowers fMAX until it achieves the specified II.

It is recommended that you use `[[intel::initiation_interval(N)]]` attribute on performance critical loops when using the `[[intel::scheduler_target_fmax_mhz(N)]]` attribute. The `[[intel::initiation_interval(N)]]` attribute takes priority over the `[[intel::scheduler_target_fmax_mhz(N)]]` attribute when they are both used on the same kernel.

### Understanding the Tutorial Design

In the tutorial, all four kernels implement the same function (BKDR Hash). The only difference is how the `[[intel::scheduler_target_fmax_mhz(N)]]` attribute and the `[[intel::initiation_interval(N)]]` attribute are applied. All specific numbers mentioned below are expected observations on Intel Arria® 10 GX FPGA. Even so, the tradeoff between fMAX and II is also expected on other devices.

In kernel `Default`, no fMAX or II constraints are provided. By default, the compiler tries to optimize throughput using heuristics to balance high fMAX and small II. The block `B1` is scheduled at less than 240 MHz, so this block is limiting this kernel's fMAX but is able to achieve II=1.

In kernel `Fmax480Attr`, the `[[intel::scheduler_target_fmax_mhz(480)]]` attribute tells the compiler to target 480 MHz. Since the II is unconstrained, the compiler inserts extra pipelining to schedule the kernel at 480 MHz instead of trying to balance fMAX and II. Now, all blocks are scheduled at the target fMAX, but block `B1` has a higher II than kernel `Default`.

In kernel `Fmax240Attr`, the `[[intel::scheduler_target_fmax_mhz(240)]]` attribute tells the compiler to target 240 MHz. Once again, all blocks are scheduled at the target fMAX, but block `B1` has a lower II than kernel `Fmax480Attr`. Since we reduce the fMAX target, the compiler inserts fewer pipeline registers in `B1` of this kernel.

In kernel `Fmax240IIAttr`, the `[[intel::scheduler_target_fmax_mhz(240)]]` attribute tells the compiler to target 240 MHz, and the `[[intel::initiation_interval(1)]]` attribute forces block `B1` to be scheduled with II=1. Since the `[[intel::initiation_interval(1)]]` attribute takes priority over the `[[intel::scheduler_target_fmax_mhz(240)]]` attribute, the compiler is not able to schedule block `B1` at the requested target fMAX but is able to achieve II=1. This achieves a similar latency as kernel `Default` but provides you the control over how much pipelining the compiler generates while still achieving the desired II on critical loops.

## Build the `scheduler_target_fmax_mhz` Tutorial

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

Locate `report.html` in the `scheduler_target_fmax_report.prj/reports/` directory.

Navigate to the Loop Analysis table (Throughput Analysis > Loop Analysis). In kernel `Default`, block `B1` is scheduled at less than 240 MHz but has II=1. In kernel `Fmax240Attr` and `Fmax480Attr`, all blocks are scheduled at the target fMAX, but they have II>1. In kernel `Fmax240IIAttr`, similar to kernel `Default`, block `B1` is scheduled at less than 240 MHz but has II=1.

Navigate to the Area Analysis of System (Area Analysis > Area Analysis of System). By comparing the results in resource usage for different fMAX and II targets, you can see that the compiler inserts more pipeline stages and therefore increases area usage if the component is scheduled for a higher fMAX.

## Run the `scheduler_target_fmax_mhz` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./scheduler_target_fmax.fpga_emu
   ```

2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./scheduler_target_fmax.fpga_sim
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./scheduler_target_fmax.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   scheduler_target_fmax.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   scheduler_target_fmax.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   scheduler_target_fmax.fpga.exe
   ```

## Example Output

```
PASSED: all kernel results are correct.
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).