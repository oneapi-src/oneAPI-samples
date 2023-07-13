
# `Speculated Iterations` Sample

This sample is an FPGA tutorial that demonstrates applying the `speculated_iterations` attribute to a loop in a task kernel to enable more efficient loop pipelining.

| Area                 | Description
|:---                  |:---
| What you will learn  | What the `speculated_iterations` attribute does. <br> How to apply the `speculated_iterations` attribute to loops in your program. <br> How to determine the optimal number of speculated iterations.
| Time to complete     | 15 minutes
| Category             | Concepts and Functionality

## Purpose

Loop speculation is an advanced loop pipelining optimization technique. It enables loop iterations to be initiated before determining whether they should have been initiated. Speculated iterations are those iterations that launch before the exit condition computation has completed. This is beneficial when the computation of the exit condition is preventing effective loop pipelining.

The `speculated_iterations` attribute is a loop attribute that enables you to directly control the number of speculated iterations for a loop.  The attribute  `[[intelfpga::speculated_iterations(N)]]` takes an integer argument `N` to specify the permissible number of iterations to speculate.

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

This sample is part of the FPGA code samples. It is categorized as a Tier 2 sample that demonstrates a compiler feature.

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

- Description of the `speculated_iterations` attribute.
- How to apply the `speculated_iterations` attribute to loops in your program.
- Optimizing the number of speculated iterations.

### Simple Example

```c++
  [[intelfpga::speculated_iterations(1)]]
  while (sycl::log10(x) < N) {
      x += 1;
  }
  dst[0] = x;
```
The loop in this example will have one speculated iteration.

### Operations with Side Effects

When launching speculated iterations, operations with side-effects (such as stores to memory) must be predicated by the exit condition to ensure functional correctness. For this reason, operations with side-effects must be scheduled until after the exit condition has been computed.

### Optimizing the Number of Speculated Iterations

Loop speculation is beneficial when the computation of the loop exit condition is the bottleneck preventing the compiler from achieving a smaller initiation interval (II). In such instances, increasing the number of speculated iterations often improves the II.  Note that this may also uncover additional bottlenecks preventing the further optimization of the loop.

However, adding speculated iterations is not without cost. They introduce overhead in nested loops, reducing overall loop occupancy. Consider the code snippet below:

```c++
for (size_t i = 0; i < kMany; ++i) {
  // The compiler may automatically infer speculated iterations
  for (size_t j = 0; complex_exit_condition(j); ++j) {
    output[i,j] = some_function(input[i,j]);
  }
}
```
The *i+1*th invocation of the inner loop cannot begin until all real and speculated iterations of its *i*th invocation have completed. This overhead is negligible if the number of speculated iterations is much less than the number of real iterations. However, when the inner loop's trip count is small on average, the overhead becomes non-negligible, and the speculated iterations can become detrimental to throughput. In such circumstances, the `speculated_iterations` attribute can be used to *reduce* the number of speculated iterations chosen by the compiler's heuristics.

In both increasing and decreasing cases, some experimentation is usually necessary. Choosing too few speculated iterations could increase the II because multiple cycles are required to evaluate the exit condition. Choosing too many speculated iterations creates unneeded "dead space" between sequential invocations of an inner loop.

### Tutorial Example

In the tutorial design's kernel, the loop's exit condition involves a logarithm and a compare operation. This complex exit condition prevents the loop from achieving ```II=1```.

The design enqueues variants of the kernel with 0, 10, and 27 speculated iterations, respectively, to demonstrate the effect of the `speculated_iterations` attribute on an Intel® Arria® 10 FPGA. Different numbers are chosen for the Intel® Stratix® 10 and Intel Agilex® 7 targets accordingly.

## Build the `Speculated Iterations` Tutorial

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

## Read the Reports

Locate `report.html` in the `speculated_iterations_report.prj/reports/` directory.

In the "Loop Analysis" section of the report, check the II of the loop in each kernel version. Use the kernel with 0 speculated iteration as a base version, check its loop II as a hint for the ideal number for speculated iterations. The information shown below is from compiling on the Intel® PAC with Intel Arria® 10 GX FPGA.

- When the number of `speculated iterations` is set to 0, loop II is 27.
- Setting the `speculated iterations` to 27 yielded an II of 1.
- Setting the `speculated iterations` to an intermediate value of 10 results in an II of 3.

These results make sense when you recall that the loop exit computation has a latency of 27 cycles (suggested by looking at loop II with 0 speculation). With no speculation, a new iteration can only be launched every 27 cycles. Increasing the speculation to 27 enables a new iteration to launch every cycle. Reducing the speculation to 10 results in an II of 3 because 10 speculated iterations multiplied by 3 cycles between iterations leave 30 cycles in which to compute the exit condition, sufficient to cover the 27-cycle exit condition.

## Run the `Speculated Iterations` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./speculated_iterations.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./speculated_iterations.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./speculated_iterations.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   speculated_iterations.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   speculated_iterations.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   speculated_iterations.fpga.exe
   ```

## Example of Output

```
Speculated Iterations: 0 -- kernel time: 8564.98 ms
Performance for kernel with 0 speculated iterations: 11675 MFLOPs
Speculated Iterations: 10 -- kernel time: 952 ms
Performance for kernel with 10 speculated iterations: 105076 MFLOPs
Speculated Iterations: 27 -- kernel time: 317 ms
Performance for kernel with 27 speculated iterations: 315181 MFLOPs
PASSED: The results are correct
```

The execution time and throughput for each kernel are displayed.

> **Note**: The performance difference will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).