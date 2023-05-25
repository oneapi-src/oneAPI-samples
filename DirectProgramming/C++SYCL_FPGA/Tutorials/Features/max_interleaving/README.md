# `max_interleaving` Sample

This sample is an FPGA tutorial that explains how to use the `max_interleaving` attribute for loops.

| Area                 | Description
|:--                   |:--
| What you will learn  | The basic usage of the `max_interleaving` attribute. <br> How the `max_interleaving` attribute affects loop resource use. <br> How to apply the `max_interleaving` attribute to loops in your program.
| Time to complete     | 15 minutes
| Category             | Concepts and Functionality

## Purpose

This tutorial demonstrates a method to reduce the area usage of inner loops that cannot realize throughput increases through interleaved execution. By default, the compiler will generate loop datapaths that enable multiple invocations of the same loop to execute simultaneously, called interleaving, in order to maximize throughput when II is greater than 1. In cases where interleaving is dynamically prohibited. For example, due to data dependency preservation, hardware resources are wasted when used to enable interleaving. The `max_interleaving` attribute can instruct the compiler to limit allocation of these hardware resources in these cases.

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

- The basic usage of the `max_interleaving` attribute.
- How the `max_interleaving` attribute affects loop throughput and resource use.
- How to apply the `max_interleaving` attribute to loops in your program.

### Description of the `max_interleaving` Attribute

Consider a pipelined inner loop `i` with an II > 1 contained inside a pipelined outer loop `j`. If the trip count of the `i` loop does not vary with respect to the iterations of the `j` loop, the compiler will automatically generate hardware that allows the `i` loop to interleave iterations from different invocations of itself. That is, each iteration of the outer `j` loop will contain a different invocation of the inner `i` loop. Each of these invocations of the `i` loop will issue iterations every II cycles, leaving II-1 cycles in between these iterations empty. These empty cycles can be used to issue iterations of a different invocation of `i` (corresponding to a different iteration of the `j` loop), which we called interleaving. Interleaving allows the generated design to have a high occupancy despite a high II, which improves throughput.

Although interleaving will generally improve throughput, there may be some scenarios where you might not want to incur the hardware cost of generating a pipelined datapath that allows interleaving. For example, in a nest of pipelined loops, pipelined iterations of an outer loop may be serialized by the compiler across an inner loop if the inner loop imposes a data dependency on the containing loop. For such scenarios, the generation of a loop datapath that supports interleaving iterations of the containing pipelined loop yields little to no benefit in throughput because the serialization to preserve the data dependency will prevent interleaving from occurring. Manually restricting or disabling the amount of interleaving on the inner loop reduces the area overhead imposed by a datapath generated to handle interleaved invocations. Users can mark a loop with the `max_interleaving` pragma to limit the number of interleaved invocations supported by the generated loop datapath.

See the following example:

```
L1: for (size_t i = 0; i < kSize; i++) {
  L2: for (size_t j = 0; j < kSize; j++) {
        temp_r[j] = SomethingComplicated(temp_a[i][j], temp_r[j]);
  }
  temp_r[i] += temp_b[i];
}
```

In this loop nest, pipelined iterations of L1 are serialized across L2 to preserve the data dependency on the array variable 'temp_r'. This means that one invocation of the `j` loop can be executing at any time; therefore, no dynamic interleaving of iterations from different invocations of the `j` loop can occur. By default, the compiler will generate a datapath that includes the capacity to run multiple interleaved iterations simultaneously. Since the data dependency prevents dynamic interleaving, the resources spent on an interleaving-capable datapath are wasted. Applying the `max_interleaving` attribute with an argument of `1` will instruct the compiler generate a datapath that restricts the interleaving capacity to a single `j` loop invocation.

## Build the `max_interleaving` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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

Locate `report.html` in the `max_interleaving_report.prj/reports/` directory.

In the "Loops analysis" view from the "Throughput Analysis" drop-down menu, in the Details pane for loop L1 (choose either Compute<0>.B6 or Compute<1>.B6 from the "Loop List" pane, then the click the "Source Location" link in the "Loop Analysis" pane):

Iteration executed serially across KernelCompute<0>.B8. Only a single loop iteration will execute inside this region due to data dependency on variable(s):

```
temp_r (max_interleaving.cpp: 56)
```
As described in the Example section above, generating a loop datapath that supports interleaving iterations of the containing pipelined loop in this scenario yields little to no benefit in throughput because preservation of the data dependency on `temp_r` requires that only one invocation of the inner loop be executed at a time. Adding the `max_interleaving(1)` attribute informs the compiler to not allocated hardware resources for interleaving on the inner loop, reducing the area overhead.

Referring to the generated report again, choose the "Summary" view, expand the "System Resource Utilization Summary" section in the Summary pane, and select the "Compile Estimated Kernel Resource Utilization Summary" subsection. In this subsection, you will see resource utilization estimates for two kernels: Compute<0> and Compute<1>. These correspond to two nearly-identical kernels, with the only difference being the `max_interleaving` attribute applied to the inner loop at line 58 taking either a 0 or 1 as its argument (`max_interleaving(0)` specifies unlimited interleaving, which is the same as the default `max_interleaving` limit when no attribute is set). Note that Compute<1>, which has restricted interleaving, uses fewer ALUTs, REGs, and MLABs than Compute<0>.

The area usage information can also be accessed on the main report page in the Summary pane. Scroll down to the section titled "System Resource Utilization Summary". Each kernel name ends in the `max_interleaving` attribute argument used for that kernel, e.g., `KernelCompute<1>` uses a `max_interleaving` attribute value of 1. You can verify that the number of ALUTs, REGs, and MLABs used for each kernel decreases when `max_interleaving` is limited to 1 compared to unlimited interleaving when `max_interleaving` is set to 0.

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

## Example Output

```
Max interleaving 0 kernel time : 0.019088 ms
Throughput for kernel with max_interleaving 0: 0.007 GFlops
Max interleaving 1 kernel time : 0.015 ms
Throughput for kernel with max_interleaving 1: 0.009 GFlops
PASSED: The results are correct
```

The stdout output shows the giga-floating point operations per second (GFlops) for each kernel.

When run on an Intel® PAC with Intel Arria® 10® 10 GX FPGA hardware board, we see that the throughput remains unchanged when using `max_interleaving(0)` or `max_interleaving(1)`. However, the kernel using `max_interleaving(1)` uses fewer hardware resources, as shown in the reports.

When run on the FPGA emulator, the `max_interleaving` attribute has no effect on runtime. You may notice that the emulator achieved higher throughput than the FPGA in this example. This anomaly occurs because this trivial example uses only a tiny fraction of the compute resources available on the FPGA.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).