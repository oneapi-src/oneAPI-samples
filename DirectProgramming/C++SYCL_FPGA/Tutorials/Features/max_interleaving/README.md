# `max_interleaving` Sample

This sample is an FPGA tutorial that explains how to use the `max_interleaving` attribute for loops.

| Area                 | Description
|:--                   |:--
| What you will learn  | The basic usage of the `max_interleaving` attribute. <br> How the `max_interleaving` attribute affects loop resource use. <br> How to apply the `max_interleaving` attribute to loops in your program.
| Time to complete     | 15 minutes
| Category             | Concepts and Functionality

## Purpose

This tutorial demonstrates a method to reduce the area usage of inner loops by disabling interleaved execution. 

When possible, the compiler will generate loop datapaths that enable multiple invocations of the same loop to execute simultaneously, called interleaving, in order to maximize throughput when II is greater than 1. 

Though the extra hardware generated is usually minor, disabling interleaving is an easy way to save area, for example, in non performance-critical paths.

The `[[intel::max_interleaving(0 or 1)]]` attribute can instruct the compiler to limit allocation of these hardware resources in these cases.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware             | Intel® Agilex® 7, Agilex® 5, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.

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

- The basic usage of the `max_interleaving` attribute.
- How the `max_interleaving` attribute affects loop throughput and resource use.
- How to apply the `max_interleaving` attribute to loops in your program.

### Description of the `max_interleaving` Attribute

#### Quick Reference
Place the `[[intel::max_interleaving(0 or 1)]]` attribute above a loop that you want to control interleaving. A parameter of `0` (the default if the attribute is not used) enables interleaving when possible, and `1` forcibly disables interleaving, even if it is possible.  

#### Detailed Explanation
Consider the following pipelined doubly nested loops: 
```cpp
float temp_r[kSize] = ...;
float temp_a[kSize * kSize] = ...;

for (int i = kSize - 1; i >= 0; i--) {
   [[intel::max_interleaving(0 or 1)]] // Controls if interleaving is enabled or disabled on the loop below
   for (int j = kSize - 1; j >= 0; j--) {
      temp_r[i] += SomethingComplicated(temp_a[i * kSize + j], temp_r[i]);
   }
}
```

Notice how `temp_r[i]` is a loop carried dependency in the inner loop. **Crucially, the dependency is with respect to the inner loop and *not* the outer loop.** The same `temp_r[i]` is updated in every inner iteration, but a different element of `temp_r` is updated in different outer iterations. As a result of the loop carried dependency with respect to the inner loop, the II of the inner loop will be very high as a new iteration of the inner loop cannot be invoked until the previous one finishes. 

Without interleaving, this is what the pipelined registers of the inner loop will look like, for II of the inner loop = 5 and II of the outer loop = 1. Each cell shows the (i, j) iteration that is currently in that stage of the inner loop:

| Cycle | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
| ---   | ---     | ---     | ---     | ---     | ---     |
| 1     | (0, 0)  |
| 2     |         | (0, 0)  |
| 3     |         |         | (0, 0)  |
| 4     |         |         |         | (0, 0)  |
| 5     |         |         |         |         | (0, 0)  |
| 6     | (0, 1)  |

Notice how the majority of the stages are empty, caused by the loop carried dependency.

With interleaving, we get the following: 

| Cycle | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
| ---   | ---     | ---     | ---     | ---     | ---     |
| 1     | (0, 0)  | 
| 2     | (1, 0)  | (0, 0)  |
| 3     | (2, 0)  | (1, 0)  | (0, 0)  |
| 4     | (3, 0)  | (2, 0)  | (1, 0)  | (0, 0)  |
| 5     | (4, 0)  | (3, 0)  | (2, 0)  | (1, 0)  | (0, 0)  |
| 6     | (0, 1)  | (4, 0)  | (3, 0)  | (2, 0)  | (1, 0)  |

Since the loop carried dependency is not with respect to the outer loop, different *invocations* of the inner loop can be pipelined into the inner loop. Notice how after an initial ramp-up period, the inner loop hardware reaches full occupancy - which will correspond to a higher throughput. 

While interleaving is desired in most situations, there may be some scenarios where you might not want to incur the hardware cost of generating a pipelined datapath that allows interleaving. 

For example, if a loop is non-performance critical and area savings are paramount, interleaving can be disabled. 

Another use case is if loop-carried memory dependencies cannot be determined at compile-time (such as dynamic array index accesses). By default, the compiler will conservatively assume interleaving may happen as to maximize throughput, and supporting hardware will be generated. If the user knows that interleaving cannot or does not frequently happen, they can manually disable interleaving. 

To disable interleaving on a loop, place `[[intel::max_interleaving(1)]]` above that loop, as shown in the example above. 


## Build the `max_interleaving` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your 'build' directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory, for example:
>
>  ```
  > C:\samples\build> cmake -G "NMake Makefiles" C:\long\path\to\code\sample\CMakeLists.txt
>  ```
### Read the Reports

Locate `report.html` in the `max_interleaving.report.prj/reports/` directory.

#### Verify That Interleaving Is Enabled/Disabled
1. Go to `Throughput Analysis` (dropdown) -> `Loop Analysis`.
2. Under `Loop List`, the 2 loops of interest are:
   1. `KernelCompute_0.inner` - inner loop of interleaving **enabled** kernel
   2. `KernelCompute_1.inner` - inner loop of interleaving **disabled** kernel
3. Find the row in the `Loop Analysis` pane corresponding to `KernelCompute_0.inner`. Notice how the II is high (>> 1) and `Max Interleaving Iterations` is greater than 1, meaning interleaving is enabled.
4. Find the row in the `Loop Analysis` pane corresponding to `KernelCompute_1.inner`. Notice how the II is high (>> 1) but the `Max Interleaving Iterations` is 1, meaning interleaving is disabled due to the attribute.

**IMPORTANT**: As mentioned above, the compiler will do some memory dependency analysis on loops. If it can determine that interleaving cannot happen, interleaving will automatically be disabled (which can be verified in the reports) and the attribute will have no effect. 

#### View the Hardware Area Savings
**NOTE**: For the most accurate numbers, you must compile to hardware so that `Quartus®` can decide where to place each hardware unit on the board.
1. Go to `Summary` (top navigation bar). In the `Summary` pane, go to `Quartus® Fitter Resource Utilization Summary`. For less accurate estimates (but you can obtain these numbers after compiling to report instead of the full hardware flow), go to `Compile Estimated Kernel Resource Utilization Summary`.
2. Verify that `KernelCompute<1>` (interleaving disabled) uses slightly fewer resources (ALMs, ALUTs, REGs, etc.) than `KernelCompute<0>` (interleaving enabled). For example, at the time of writing this tutorial, this is the final resource usage when compiling for the Intel® FPGA SmartNIC N6001-PL:

|                 | ALM  | ALUT | REG   | MLAB | RAM | DSP |
| ---             | ---  | ---  | ---   | ---  | --- | --- |
| KernelCompute_0 | 1703 | 3406 | 7769  | 34   | 100 | 6   |
| KernelCompute_1 | 1653 | 3307 | 6741  | 33   | 100 | 6   | 

## Run the `max_interleaving` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./max_interleaving.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./max_interleaving.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./max_interleaving.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   max_interleaving.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   max_interleaving.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   max_interleaving.fpga.exe
   ```

## Example Output On FPGA Hardware

```
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)
Max interleaving 0 kernel time : 0.062976 ms
Throughput for kernel with max_interleaving 0: 1.041 GFlops
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)
Max interleaving 1 kernel time : 0.909 ms
Throughput for kernel with max_interleaving 1: 0.072 GFlops
PASSED: The results are correct
```

The stdout output shows the giga-floating point operations per second (GFlops) for each kernel.

When run on the Intel® FPGA SmartNIC N6001-PL, we see that the throughput is significantly higher for `max_interleaving(0)` (interleaving enabled) than `max_interleaving(1)`, showing the effectiveness of interleaving. However, the kernel using `max_interleaving(1)` uses slightly fewer hardware resources, as shown in the reports. 

While the throughput differences are substantial, if the interleaving loops were a small part of a kernel whose total runtime was an order of magnitude greater than these loops, it may be worth it to disable interleaving for hardware savings.

When run on the FPGA emulator, the `max_interleaving` attribute has no effect on runtime. Additionally, the emulator may sometimes achieve higher throughput than the FPGA. This is especially true if the kernel uses a tiny fraction of the compute resources available on the FPGA.

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
