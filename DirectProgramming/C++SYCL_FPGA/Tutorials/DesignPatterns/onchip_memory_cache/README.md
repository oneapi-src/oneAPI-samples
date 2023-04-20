# `On-Chip Memory Cache` Sample

This FPGA tutorial demonstrates how to build a simple cache (implemented in FPGA registers) to store recently-accessed memory locations so that the compiler can achieve II=1 on critical loops in task kernels.

| Area                 | Description
|:--                   |:--
| What you will learn  | How and when to implement the on-chip memory cache optimization
| Time to complete     | 30 minutes
| Category             | Code Optimization

## Purpose

This sample demonstrates the following concepts:

- How to implement the on-chip memory cache optimization technique
- The scenarios in which this technique benefits performance
- How to tune the cache depth

## Prerequisites

| Optimized for                     | Description
---                                 |---
| OS                                | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, you must have Intel® Quartus® Prime Pro Edition and one of the following simulators installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples. It is categorized as a Tier 3 sample that demonstrates a design pattern.

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

In SYCL* task kernels for FPGA, our objective is to achieve an initiation interval (II) of 1 on performance-critical loops. This means that a new loop iteration is launched on every clock cycle, maximizing the loop's throughput.

When the loop contains a loop-carried variable implemented in on-chip memory, the compiler often *cannot* achieve II=1 because the memory access takes more than one clock cycle. If the updated memory location may be needed on the next loop iteration, the next iteration must be delayed to allow time for the update, hence II > 1.

The technique of using the `OnchipMemoryWithCache` class breaks this dependency by storing recently-accessed values in a cache capable of a 1-cycle read-modify-write operation. The cache is implemented in FPGA registers rather than on-chip memory. By pulling memory accesses preferentially from the register cache, the loop-carried dependency is broken.

### Determining When the On-Chip Memory with Cache Technique is Appropriate

**Failure to achieve II=1 because of a loop-carried memory dependency in on-chip memory**: The on-chip memory with cache technique is applicable if the compiler could not pipeline a loop with II=1 because of an on-chip memory dependency. (If the compiler could not achieve II=1 because of a **global** memory dependency, this technique does not apply as the access latencies are too great.)

To check this for a given design, view the *Loops Analysis* section of its optimization report. The report lists the II of all loops and explains why a lower II is not achievable. Check whether the reason given resembles "the compiler failed to schedule this loop with smaller II due to memory dependency". The report will describe the "most critical loop feedback path during scheduling". Check whether this includes on-chip memory load/store operations on the critical path.

**An II=1 loop with a load operation of latency 1**: The compiler can reduce the latency of on-chip memory accesses to achieve II=1. In doing so, the compiler makes a trade-off by sacrificing f<sub>MAX</sub> to improve the II.

In a design with II=1 critical loops but lower than desired f<sub>MAX</sub>, the on-chip memory with cache technique may still be applicable. It can help recover f<sub>MAX</sub> by enabling the compiler to achieve II=1 with a higher latency memory access.

To check whether this is the case for a given design, view the "Kernel Memory Viewer" section of the optimization report. Select the on-chip memory of interest from the Kernel Memory List, and mouse over the load operation "LD" to check its latency. If the latency of the load operation is 1, this is a clear sign that the compiler has attempted to sacrifice f<sub>MAX</sub> to improve loop II.

### Implementing the On-Chip Memory with Cache Technique

The tutorial demonstrates the technique using a program that computes a histogram. The histogram operation accepts an input vector of values, separates the values into groups, and counts the number of values per group. For each input value, an output group is determined, and the count for that group is incremented. This count is stored in the on-chip memory, and the increment operation requires reading from memory, performing the increment, and storing the result. This read-modify-write operation is the critical path that can result in II > 1.

To reduce II, the idea is to store recently-accessed values in an FPGA register-implemented cache capable of a 1-cycle read-modify-write operation. If the memory location required on a given iteration exists in the cache, it is pulled from there. The updated count is written back to the cache. The cache is implemented as a shift registe, which stores the N most recent writes.  As a value exits the end of the shift register, it is written into the on-chip memory. The implementation of the cache is hidden away inside the OnchipMemoryWithCache class, which users may reuse in their own applications.

### Selecting the Cache Depth

While any value of `CACHE_DEPTH` results in functional hardware, finding the ideal value of `CACHE_DEPTH` requires experimentation. The depth of the cache needs to cover the latency of the on-chip memory access. To determine the correct value, start with a value of 1 and then increase it until both II = 1 and load latency > 1. In this tutorial, a `CACHE_DEPTH` of 7 is needed to achieve this goal.

For user designs, each iteration takes only a few moments to compile the reports. It is important to find the **minimal** value of `CACHE_DEPTH` that results in a maximal performance increase. Unnecessarily large values of `CACHE_DEPTH` consume FPGA resources unnecessarily and can reduce f<sub>MAX</sub>; therefore, at a `CACHE_DEPTH` that results in II=1 and load latency = 1, if further increases to `CACHE_DEPTH` show no improvement, `CACHE_DEPTH` should not be increased any further.

This tutorial creates multiple kernels sweeping across different cache depths within a single design. This approach allows a single compile of the reports to determine the optimal cache depth.

## Build the `On-Chip Memory Cache` Tutorial

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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

    1. Compile for emulation (fast compile time, targets emulated FPGA device):
        ```
        make fpga_emu
        ```
    2. Generate the optimization report:
        ```
        make report
        ```
       The report resides at `onchip_memory_cache_report.prj/reports/report.html`.

       Compare the Loop Analysis reports for kernels with various cache depths, as described in the "When is the on-chip memory cache technique applicable?" section.  This will illustrate that any cache depth > 0 allows a loop II of 1.

       Open the Kernel Memory viewer and compare the Load Latency on the loads from kernels with various cache depths, as describe in the "When is the on-chip memory cache technique applicable?" section. This will illustrate that a cache depth of at least 7 is required to achieve a load latency of > 1.

    3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
        ```
        make fpga_sim
        ```
    4. Compile for FPGA hardware (longer compile time, targets FPGA device):
        ```
        make fpga
        ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Agilex 7™ device family, which is the default.
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

   1. Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report:
      ```
      nmake report
      ```
       The report resides at `onchip_memory_cache_report.prj.a/reports/report.html`.

       Compare the Loop Analysis reports for kernels with various cache depths, as described in the "When is the on-chip memory cache technique applicable?" section.  This will illustrate that any cache depth > 0 allows a loop II of 1.

       Open the Kernel Memory viewer and compare the Load Latency on the loads from kernels with various cache depths, as describe in the "When is the on-chip memory cache technique applicable?" section. This will illustrate that a cache depth of at least 7 is required to achieve a load latency of > 1.

   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `On-Chip Memory Cache` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./onchip_memory_cache.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./onchip_memory_cache.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./onchip_memory_cache.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   onchip_memory_cache.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   onchip_memory_cache.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   onchip_memory_cache.fpga.exe
   ```

## Example Output

```
Platform name: Intel(R) FPGA SDK for OpenCL(TM)
Device name: pac_s10 : Intel PAC Platform (pac_f100000)


Number of inputs: 16777216
Number of outputs: 64

Beginning run with cache depth 0 (no cache)
Data check succeeded for cache depth 0
Kernel execution time: 0.120812 seconds
Kernel throughput for cache depth 0: 529.749056 MB/s

Beginning run with cache depth 1
Data check succeeded for cache depth 1
Kernel execution time: 0.060410 seconds
Kernel throughput for cache depth 1: 1059.430983 MB/s

Beginning run with cache depth 2
Data check succeeded for cache depth 2
Kernel execution time: 0.060408 seconds
Kernel throughput for cache depth 2: 1059.466199 MB/s

Beginning run with cache depth 3
Data check succeeded for cache depth 3
Kernel execution time: 0.060410 seconds
Kernel throughput for cache depth 3: 1059.429632 MB/s

Beginning run with cache depth 4
Data check succeeded for cache depth 4
Kernel execution time: 0.060407 seconds
Kernel throughput for cache depth 4: 1059.476214 MB/s

Beginning run with cache depth 5
Data check succeeded for cache depth 5
Kernel execution time: 0.060409 seconds
Kernel throughput for cache depth 5: 1059.447398 MB/s

Beginning run with cache depth 6
Data check succeeded for cache depth 6
Kernel execution time: 0.060409 seconds
Kernel throughput for cache depth 6: 1059.445539 MB/s

Beginning run with cache depth 7
Data check succeeded for cache depth 7
Kernel execution time: 0.060407 seconds
Kernel throughput for cache depth 7: 1059.484369 MB/s

Beginning run with cache depth 8
Data check succeeded for cache depth 8
Kernel execution time: 0.060409 seconds
Kernel throughput for cache depth 8: 1059.438594 MB/s

Verification PASSED
```

### Understanding the Results

As the sample results above demonstrate, adding the cache to achieve an II of 1 approximately doubles the throughput of the kernel.

Because the f<sub>MAX</sub> of a design is determined by the slowest kernel, we are not able to see the f<sub>MAX</sub> improvement of increasing the load latency from 1 to 4. To see that, you would have to compile the design with a single kernel at a time.

When caching is used, performance increases. As previously mentioned, this technique should result in an II reduction, which should lead to a throughput improvement. The technique can also improve f<sub>MAX</sub> if the compiler had previously implemented a latency=1 load operation, in which case the f<sub>MAX</sub> increase should result in a further throughput improvement.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
