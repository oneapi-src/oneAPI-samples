# `Remove Loop Carried Dependency` Sample

This tutorial demonstrates how to remove a loop-carried dependency to improve the performance of the FPGA device code.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to remove loop carried dependencies from your FPGA device code and when to apply the method.
| Time to complete       | 25 minutes
| Category               | Code Optimization

## Purpose

This tutorial sample demonstrates the following concepts:

- The impact of loop carried-dependencies on FPGA SYCL kernel performance
- An optimization technique to break loop-carried data dependencies in critical loops

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstrates a design pattern.

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

This tutorial demonstrates how to remove a loop-carried dependency in FPGA device code.

A snippet of the baseline unoptimized code (the `Unoptimized` function in `src/loop_carried_dependency.cpp`) is shown below.

```
double sum = 0;
for (size_t i = 0; i < N; i++) {
  for (size_t j = 0; j < N; j++) {
    sum += a[i * N + j];
  }
  sum += b[i];
}
result[0] = sum;
```
In the unoptimized kernel, a sum is computed over two loops.  The inner loop sums over the `a` data and the outer loop over the `b` data. Since the value `sum` updates in both loops, this introduces a _loop carried dependency_ that causes the outer loop to be serialized. It allows only one invocation of the outer loop to be active at a time, which reduces performance.

A snippet of the optimized code (the `Optimized` function in `src/loop_carried_dependency.cpp`) is given below, which removes the loop carried dependency on the `sum` variable.

```
double sum = 0;

for (size_t i = 0; i < N; i++) {
  // Step 1: Definition
  double sum_2 = 0;

  // Step 2: Accumulation of array A values for one outer loop iteration
  for (size_t j = 0; j < N; j++) {
    sum_2 += a[i * N + j];
  }

  // Step 3: Addition of array B value for an outer loop iteration
  sum += sum_2;
  sum += b[i];
}

result[0] = sum;
```

The optimized kernel demonstrates the use of an independent variable `sum_2` that is not updated in the outer loop and removes the need to serialize the outer loop, which improves the performance.

### When to Use This Technique

Look at the _Compiler Report > Throughput Analysis > Loop Analysis_ section in the reports. The report lists the II and details for each loop. The technique presented in this tutorial may be applicable if the _Brief Info_ of the loop shows _Serial exe: Data dependency_.  The details pane may provide more information.

```
* Iteration executed serially across _function.block_. Only a single loop iteration will execute inside this region due to data dependency on variable(s):
    * sum (_filename:line_)
```

## Build the `Remove Loop Carried Dependency` Sample

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
   2. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `loop_carried_dependency_report.prj/reports/report.html`.

      (Optional) Navigate to the _Loops Analysis_ view of the report (under _Throughput Analysis_) and observe that the loop in block `UnOptKernel.B1` is showing _Serial exe: Data dependency_.  Click the *source location* field in the table to see the details for the loop. It should show 1 as the maximum interleaving iteration of the loop, as the loop is serialized. Now, observe that the loop in block `OptKernel.B1` is not marked as *Serialized*. It shows 12 as the maximum Interleaving iterations of the loop.

   3. Compile for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
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
   2. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `loop_carried_dependency.prj.a/reports/report.html`.

      (Optional) Navigate to the _Loops Analysis_ view of the report (under _Throughput Analysis_) and observe that the loop in block `UnOptKernel.B1` is showing _Serial exe: Data dependency_.  Click the *source location* field in the table to see the details for the loop. It should show 1 as the maximum interleaving iteration of the loop, as the loop is serialized. Now, observe that the loop in block `OptKernel.B1` is not marked as *Serialized*. It shows 12 as the maximum Interleaving iterations of the loop.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your `build` directory in a shorter path, for example `C:\samples\build`. You can then build the sample in the new location, but you must specify the full path to the build files.

## Run the `Remove Loop Carried Dependency` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./loop_carried_dependency.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./loop_carried_dependency.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./loop_carried_dependency.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   loop_carried_dependency.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   loop_carried_dependency.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   loop_carried_dependency.fpga.exe
   ```

## Example Output

### Example Output on FPGA Device

>**Note**: In the sample, applying the optimization yields a total execution time reduction by almost a factor of 4. The Initiation Interval (II) for the inner loop is 12 because a double floating point add takes 11 cycles on the FPGA.

```
Number of elements: 16000
Run: Unoptimized:
kernel time : 10685.3 ms
Run: Optimized:
kernel time : 2736.47 ms
PASSED
```

### Example Output on FPGA Emulation

```
Number of elements: 16000

Emulator output does not demonstrate true hardware performance. The design may need to run on actual hardware to observe the performance benefit of the optimization exemplified in this tutorial.

Run: Unoptimized:
kernel time : 334.33 ms
Run: Optimized:
kernel time : 335.345 ms
PASSED
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
