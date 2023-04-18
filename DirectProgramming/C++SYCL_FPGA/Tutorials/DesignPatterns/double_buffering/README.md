# `Double Buffering` Sample

This FPGA tutorial demonstrates how to parallelize host-side processing and buffer transfers between host and device with kernel execution, which can improve overall application performance.

| Area                 | Description
|:---                  |:---
| What you will learn  | How and when to implement the double buffering optimization technique
| Time to complete     | 30 minutes
| Category             | Code Optimization

## Purpose

This sample demonstrates double buffering to overlap kernel execution with buffer transfers and host processing. In an application where the FPGA kernel is executed multiple times, the host must perform the following processing and buffer transfers before each kernel invocation.

- The **output data** from the *previous* invocation must be transferred from the device to the host and then processed by the host. Examples of this processing include copying the data to another location, rearranging the data, and verifying it in some way.

- The **input data** for the *next* invocation must be processed by the host and then transferred to the device. Examples of this processing include copying the data from another location, rearranging the data for kernel consumption, and generating the data in some way.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 2 sample that demonstrates a design pattern.

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware           | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software           | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Key Implementation Details

The key concepts discussed in this sample are as followed:

- The double buffering optimization technique
- Determining when double buffering is beneficial
- How to measure the impact of double buffering

## Build the `Double Buffering` Sample

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

### On Linux*

1. Change to the sample directory.
2. Build the program for the Agilex® 7 device family, which is the default.

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

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   3. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `double_buffering_report.prj/reports/report.html`. Note that because the optimization occurs at the *runtime* level, the FPGA compiler report will not show a difference between the optimized and unoptimized cases.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Agilex® 7 device family, which is the default.
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
   2. Compile for simulation (fast compile time, targets simulated FPGA device).
      ```
      nmake fpga_sim
      ```
   3. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `double_buffering_report.prj.a/reports/report.html`. Note that because the optimization occurs at the *runtime* level, the FPGA compiler report will not show a difference between the optimized and unoptimized cases.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your `build` directory in a shorter path, for example `C:\samples\build`. You can then build the sample in the new location, but you must specify the full path to the build files.

## Run the `Double Buffering` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./double_buffering.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./double_buffering.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./double_buffering.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   double_buffering.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   double_buffering.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   double_buffering.fpga.exe
   ```

## Example Output

### Example Output for an FPGA Device

```
Platform name: Intel(R) FPGA SDK for OpenCL(TM)
Device name: pac_a10 : Intel PAC Platform (pac_ee00000)


Executing kernel 100 times in each round.

*** Beginning execution, without double buffering
Launching kernel #0
Launching kernel #10
Launching kernel #20
Launching kernel #30
Launching kernel #40
Launching kernel #50
Launching kernel #60
Launching kernel #70
Launching kernel #80
Launching kernel #90

Overall execution time without double buffering = 29742 ms
Total kernel-only execution time without double buffering = 17856 ms
Throughput = 35.255249 MB/s


*** Beginning execution, with double buffering.
Launching kernel #0
Launching kernel #10
Launching kernel #20
Launching kernel #30
Launching kernel #40
Launching kernel #50
Launching kernel #60
Launching kernel #70
Launching kernel #80
Launching kernel #90

Overall execution time with double buffering = 17967 ms
Total kernel-only execution time with double buffering = 17869 ms
Throughput = 58.35976 MB/s


Verification PASSED
```

### Example Output for the FPGA Emulator

```
Emulator output does not demonstrate true hardware performance. The design may need to run on actual hardware to observe the performance benefit of the optimization exemplified in this tutorial.

Platform name: Intel(R) FPGA Emulation Platform for OpenCL(TM)
Device name: Intel(R) FPGA Emulation Device


Executing kernel 20 times in each round.

*** Beginning execution, without double buffering
Launching kernel #0
Launching kernel #10

Overall execution time without double buffering = 56 ms
Total kernel-only execution time without double buffering = 3 ms
Throughput = 5.7965984 MB/s


*** Beginning execution, with double buffering.
Launching kernel #0
Launching kernel #10

Overall execution time with double buffering = 6 ms
Total kernel-only execution time with double buffering = 2 ms
Throughput = 47.919624 MB/s


Verification PASSED

```

## `Double Buffering` Guided Design Walkthrough

### Determining When Double Buffering Is Possible

Without double buffering, host processing and buffer transfers occur *between* kernel executions; therefore, there is a gap in time between kernel executions, which you can refer to as kernel *downtime* (see the image below). If these operations overlap with kernel execution, the kernels can execute back-to-back with minimal downtime increasing overall application performance.

![](assets/downtime.png)

Before discussing the concepts, we must first define the required variables.

| Variable | Description
|:---      |:---
| **R**    | Time to transfer the kernel output buffer from device to host
| **Op**   | Host-side processing time of kernel output data (**output processing**)
| **Ip**   | Host-side processing time for kernel input data (**input processing**)
| **W**    | Time to transfer the kernel input buffer from host to device
| **K**    | Kernel execution time

In general, **R**, **Op**, **Ip**, and **W** operations must all complete before the next kernel is launched. To maximize performance, while one kernel is executing on the device, these operations should execute simultaneously on the host and operate on a second set of buffer locations. They should complete before the current kernel completes, allowing the next kernel to be launched immediately with no downtime. In general, to maximize performance, the host must launch a new kernel every **K**.

This leads to the following constraint to minimize kernel downtime: **R** + **Op** + **Ip** + **W** <= **K**.

If the above constraint is not satisfied, a performance improvement may still be observed because *some* overlap (perhaps not complete overlap) is still possible. Further improvement is possible by extending the double buffering concept to N-way buffering (see the corresponding tutorial).

### Measuring the Impact of Double Buffering

You must get a sense of the kernel downtime to identify the degree to which this technique can help improve performance.

This can be done by querying the total kernel execution time from the runtime and comparing it to the overall application execution time. In an application where kernels execute with minimal downtime, these two numbers will be close. However, if kernels have a significant downtime, the overall execution time will notably exceed kernel execution time. The tutorial code exemplifies how to do this.

### Implementation Notes

The basic implementation flow is as follows:

1. Perform the input processing for the first two kernel executions and queue them both.
2. Call the `process_output()` method immediately (automatically blocked by the SYCL* runtime) on the first kernel completing because of the implicit data dependency.
3. When the first kernel completes, the second kernel begins executing immediately because it was already queued.
4. While the second kernel runs, the host processes the output data from the first kernel and prepares the third kernel's input data.
5. As long as the above operations complete before the second kernel completes, the third kernel is queued early enough to allow it to be launched immediately after the second kernel.
6. Repeat the process.

### Impact of Double Buffering

A test compile of this tutorial design achieved a maximum frequency (f<sub>MAX</sub>) of approximately 340 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results with and without double buffering are shown in the following table:

| Configuration             | Overall Execution Time (ms)  | Total Kernel Execution time (ms)
|:--                        |:--                           |:--
| Without double buffering  | 23462                        | 15187
| With double buffering     | 15145                        | 15034

In both runs, the total kernel execution time is similar as expected; however, without double buffering, the overall execution time exceeds the total kernel execution time, implying there is downtime between kernel executions. With double buffering, the overall execution time is close to the total kernel execution time.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
