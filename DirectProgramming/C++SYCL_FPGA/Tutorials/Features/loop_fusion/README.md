# `loop_fusion` Sample

This sample is an FPGA tutorial that demonstrates how loop fusion is used and how it affects performance.

| Area                 | Description
|:--                   |:--
| What you will learn  | Basics of loop fusion<br/>The reasons for loop fusion<br/>How to use loop fusion to increase performance<br/>Understanding safe application of loop fusion
| Time to complete     | 20 minutes
| Category             | Concepts and Functionality


## Purpose

This sample demonstrates how to apply loop fusion to loops in your design. It is necessary to understand the motivation and consequences of loop fusion in your design.

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

The sample illustrates the important concepts.

- The basics of loop fusion.
- The reasons for loop fusion.
- How to use loop fusion to increase performance.
- Understanding safe application of loop fusion.

The file `loop_fusion.cpp` contains four kernels, all of which contain an outer loop and two inner loops.

| Kernel Name            | Description
|:---                    |:---
|`DefaultFusionKernel`   | This kernel contains two inner loops with equal trip counts which fuse by default.
|`NoFusionKernel`        | This kernel has two inner loops with equal trip counts as in `DefaultFusionKernel`, but the compiler is instructed not to fuse the loops using the `intel::nofusion` attribute.
|`DefaultNoFusionKernel` | This kernel contains two inner loops with unequal trip counts, which the compiler does not fuse by default.
|`FusionFunctionKernel`  | This kernel contains two inner loops with unequal trip counts as in `DefaultNoFusionKernel`, but the compiler is instructed to fuse the loops using the `fpga_loop_fuse<N>(f)` function.

### Loop Fusion

Loop fusion is a compiler transformation in which adjacent loops are merged into a single loop over the same index range. This transformation is typically applied to reduce loop overhead and improve runtime performance. Loop control structures can represent a significant area overhead on designs produced by the compiler. Fusing two loops into one loop reduces the number of required loop-control structures, which reduces overhead.

In addition, fusing outer loops can introduce concurrency where there was previously none. Consider two adjacent loops L<sub>j</sub> and L<sub>k</sub>. Within each loop, independent operations can be run concurrently, but concurrency cannot be attained <i>across</i> the loops. Combining the bodies of L<sub>j</sub> and L<sub>k</sub> forms a single loop L<sub>f</sub> with a body that spans the bodies of L<sub>j</sub> and L<sub>k</sub>. In the combined loops, concurrency can be attained for independent instructions that were formerly in separate loops. In effect, the two loops now execute as one in L<sub>f</sub> in a lockstep fashion, providing possible latency improvements.

Loop fusion joins loops at the same nesting level. The merging of nested loops is known as *loop coalescing*, and tools to achieve this are described in the documentation and in the [`loop_coalesce` code sample](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/loop_coalesce).

#### Default Loop Fusion

The compiler attempts to fuse adjacent loops by default when profitable and when memory dependencies allow. If only one of the two loops has stall-free logic or if only one of the two loops is tagged with the `intel::ivdep` attribute, the compiler will not fuse loops by default when two adjacent loops have unequal trip counts.

#### Explicit Loop Fusion

The case when there are two adjacent loops L<sub>j</sub> and L<sub>k</sub>, and iteration *m* of L<sub>k</sub> depends on iteration *n* > *m* of L<sub>j</sub> is known as a *negative-distance dependency*.  A negative-distance dependency cannot be fulfilled when loop fusion is performed. The compiler will not fuse loops that are believed to have a negative-distance dependency, even when the `fpga_loop_fuse<N>(f)` function is used.

Compiler loop fusion profitability and legality heuristics can be overridden using the `fpga_loop_fuse<N>(f)` function. The `fpga_loop_fuse<N>(f)` function takes a function `f` containing loops, and an optional unsigned template parameter `N`, which specifies the number of nesting depths in which fusion should be performed. The default number of nesting depths is `N=1`.

For example, consider a function `f` containing the following loops:
```c++
for (...) { // L_11
  for (...) { // L_12
    // ...
  }
}
for (...) { // L_21
  for (...) { // L_22
    // ...
  }
}
```
When `N=1`, `fpga_loop_fuse<N>(f)` tells the compiler to fuse L<sub>11</sub> with L<sub>21</sub>, but not L<sub>12</sub> with L<sub>22</sub>. When `N=2`, `fpga_loop_fuse<N>(f)` tells the compiler to fuse L<sub>11</sub> with L<sub>21</sub>, and L<sub>12</sub> with L<sub>22</sub>.

#### Overriding Compiler Memory Checks

The compiler may conservatively not fuse a pair of loops due to a suspected memory dependency when such a dependency may not exist. In this situation, the compiler can be told to ignore memory safety checks by using the `fpga_loop_fuse_independent<N>(f)` function. This function requires the same parameters as the `fpga_loop_fuse<N>(f)` function.

>**Important**: Functional incorrectness may result if `fpga_loop_fuse_independent<N>(f)` is applied where a negative-distance dependency exists.

## Build the `Loop Fusion` Tutorial

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
Locate `report.html` in the `loop_fusion_report.prj/reports/` directory.

Navigate to the Loops Analysis section of the optimization report under Throughput Analysis and notice that two loops were fused to one in both `DefaultFusionKernel` and in `FusionFunctionKernel`, but not in `NoFusionKernel` or in `DefaultNoFusionKernel`.

In both cases where fusion has occurred, the number of loop cycles has decreased, since the total number of loop iterations decreased due to loop fusion, while the II, speculated iterations and latency are the same in the fused and non-fused loops.

Navigate to the Area Analysis of the system under Area Analysis. The Kernel System section displays the area consumption of each kernel. Notice the area savings when loop fusion is performed in`DefaultFusionKernel`, against when it is off in `NoFusionKernel`.  As well, notice the area savings when loop fusion is manually turned on in`FusionFunctionKernel`, against when it is off by default in `DefaultNoFusionKernel`.

## Run the `Loop Fusion` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./loop_fusion.fpga_emu
   ```
2. Run the sample on the FPGA simulator device (the kernel executes on the CPU).
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./loop_fusion.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./loop_fusion.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   loop_fusion.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device (the kernel executes on the CPU).
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   loop_fusion.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   loop_fusion.fpga.exe
   ```

## Example Output

```
Throughput for kernel with default loop fusion and with equally-sized loops: 1.48999 Ops/ns
Throughput for kernel with the nofusion attribute and with equally-sized loops: 0.745144 Ops/ns
Throughput for kernel without fusion by default with unequally-sized loops: 0.745192 Ops/ns
Throughput for kernel with a loop fusion function with unequally-sized loops: 1.49017 Ops/ns
PASSED: The results are correct
```

Loop fusion increases the throughput by ~100% in both the cases with equally-sized and unequally-sized loops.

>**Note**: This performance difference will be apparent only when running on FPGA hardware. The emulator and simulator, while useful for verifying functionality, will generally not reflect differences in performance.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).