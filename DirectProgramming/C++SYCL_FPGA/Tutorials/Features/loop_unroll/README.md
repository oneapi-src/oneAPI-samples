# `Loop Unroll` Sample

This sample is a tutorial that demonstrates a simple example of unrolling loops to improve throughput for a SYCL*-compliant FPGA program.

| Area                 | Description
|:--                   |:--
| What you will learn  |  Basics of loop unrolling. <br> How to unroll loops in your program. <br> Determining the optimal unroll factor for your program.
| Time to complete     | 15 minutes
| Category             | Concepts and Functionality

## Purpose

The loop unrolling mechanism is used to increase program parallelism by duplicating the compute logic within a loop. The *unroll factor* is the number of times the loop logics is duplicated. Whether the *unroll factor* is equal to the number of loop iterations loop unroll methods can be categorized as *full-loop unrolling* and *partial-loop unrolling*.

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

- Understanding the basics of loop unrolling.
- Showing how to unroll loops in your program.
- Determining the optimal unroll factor for your program.

### Example: Full-Loop Unrolling

```c++
// Before unrolling loop
#pragma unroll
for(i = 0 ; i < 5; i++){
  a[i] += 1;
}

// Equivalent code after unrolling
// There is no longer any loop
a[0] += 1;
a[1] += 1;
a[2] += 1;
a[3] += 1;
a[4] += 1;
```

A full unroll is a special case where the unroll factor is equal to the number of loop iterations. Here, the compiler instantiates five adders instead of one adder.

### Example: Partial-Loop Unrolling

```c++
// Before unrolling loop
#pragma unroll 4
for(i = 0 ; i < 20; i++){
  a[i] += 1;
}

// Equivalent code after unrolling by a factor of 4
// The resulting loop has five (20 / 4) iterations
for(i = 0 ; i < 5; i++){
  a[i * 4] += 1;
  a[i * 4 + 1] += 1;
  a[i * 4 + 2] += 1;
  a[i * 4 + 3] += 1;
}
```

Each loop iteration in the "equivalent code" contains four unrolled invocations of the first. The compiler instantiates four adders instead of one adder. Because there is no data dependency between iterations in the loop, the compiler schedules all four adds in parallel.

### Determining the Optimal Unroll Factor

In an FPGA design, unrolling loops is a common strategy to directly trade-off on-chip resources for increased throughput. When selecting the unroll factor for a specific loop, the intent is to improve throughput while minimizing resource utilization. It is also important to be mindful of other throughput constraints in your system, such as memory bandwidth.

### Tutorial Design

This tutorial demonstrates this trade off with a simple vector add kernel. The tutorial shows how increasing the unroll factor on a loop increases throughput until another bottleneck is encountered. This example is constructed to run up against global memory bandwidth constraints.

For this example, let us consider the Intel® Programmable Acceleration Card with Intel Arria® 10 GX FPGA. The memory bandwidth of this FPGA board is about 6 GB/second. The tutorial design will likely run at around 300 MHz when targeting this BSP. In this design, the FPGA design processes a new iteration every cycle in a pipeline-parallel fashion. The theoretical computation limit for one adder is:

- **GFlops**: 300 MHz \* 1 float = 0.3 GFlops
- **Computation Bandwidth**: 300 MHz \* 1 float * 4 Bytes   = 1.2 GB/s

You repeat this back-of-the-envelope calculation for different unroll factors:

|Unroll Factor  | GFlops (GB/s) | Computation Bandwidth (GB/s)
|:---           |:---           |:---
|1              | 0.3           | 1.2
|2              | 0.6           | 2.4
|4              | 1.2           | 4.8
|8              | 2.4           | 9.6
|16             | 4.8           | 19.2

On an Intel® Programmable Acceleration Card with Intel Arria® 10 GX FPGA, you can reasonably predict that the program will become memory-bandwidth limited when the unroll factor grows from 4 to 8. If you have access to such an FPGA board, check this prediction by running the design following the instructions below (providing the appropriate BSP when running `cmake`).

## Build the `Loop Unroll` Tutorial

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
Locate `report.html` in the `loop_unroll_report.prj/reports/` directory.

Navigate to the Area Report and compare the kernels' FPGA resource utilization with unroll factors of 1, 2, 4, 8, and 16. In particular, check the number of DSP resources consumed. You should see that the area grows roughly linearly with the unroll factor.

You can also check the achieved system f<sub>MAX</sub> to verify the earlier calculations.

## Run the `Loop Unroll` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./loop_unroll.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./loop_unroll.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./loop_unroll.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   loop_unroll.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   loop_unroll.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   loop_unroll.fpga.exe
   ```

## Example Output

```
Input Array Size:  67108864
UnrollFactor 1 kernel time : 255.749 ms
Throughput for kernel with UnrollFactor 1: 0.262 GFlops
UnrollFactor 2 kernel time : 140.285 ms
Throughput for kernel with UnrollFactor 2: 0.478 GFlops
UnrollFactor 4 kernel time : 68.296 ms
Throughput for kernel with UnrollFactor 4: 0.983 GFlops
UnrollFactor 8 kernel time : 44.567 ms
Throughput for kernel with UnrollFactor 8: 1.506 GFlops
UnrollFactor 16 kernel time : 39.175 ms
Throughput for kernel with UnrollFactor 16: 1.713 GFlops
PASSED: The results are correct
```

The following table summarizes the execution time (in ms), throughput (in GFlops), and number of DSPs used for unroll factors of 1, 2, 4, 8, and 16 for a default input array size of 64M floats (2 ^ 26 floats) on Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA:

Unroll Factor  | Kernel Time (ms) | Throughput (GFlops) | Num of DSPs
|:---          |:---              |:---                 |:---
|1             | 242              | 0.277               | 1
|2             | 127              | 0.528               | 2
|4             | 63               | 1.065               | 4
|8             | 46               | 1.459               | 8
|16            | 44               | 1.525               | 16

Notice that when the unroll factor increases from 1 to 2 and from 2 to 4, the kernel execution time decreases by a factor of two. Correspondingly, the kernel throughput doubles. However, when the unroll factor is increased from 4 to 8 or from 8 to 16, the throughput no longer scales by a factor of two at each step. The design is now bound by memory bandwidth limitations instead of compute unit limitations, even though the hardware is replicated.

>**Note**: These performance differences will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).