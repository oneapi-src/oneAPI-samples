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

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware           | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software           | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel® DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

### Additional Documentation

- *[Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html)* helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- *[FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)* helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)* helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

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

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Remove Loop Carried Dependency` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
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

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the *[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```
   For **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   For a custom FPGA platform, ensure that the board support package is installed on your system then enter a command similar to the following:

   ```
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

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

    (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_carried_dependency.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_carried_dependency.fpga.tar.gz).


### On Windows*

>**Note**: The Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) does not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   For a custom FPGA platform, ensure that the board support package is installed on your system then enter a command similar to the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DUSM_HOST_ALLOCATIONS_ENABLED=1
   ```

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

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Run the `Remove Loop Carried Dependency` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./loop_carried_dependency.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   ./loop_carried_dependency.fpga_sim
   ```
3. Run the sample on the FPGA device.
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
   loop_carried_dependency.fpga_sim.exe
   ```
3. Run the sample on the FPGA device.
   ```
   loop_carried_dependency.fpga.exe
   ```

### Build and Run the Samples on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.

Use the Linux instructions to build and run the program.

You can specify an FPGA runtime node using a single line script similar to the following example.

```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:fpga_runtime:ppn=2` (lower case L) assigns one full node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes           |Command Options
  |:---                      |:---
  |FPGA Compile Time         |`qsub -I -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime (Stratix 10) |`qsub -I -l nodes=1:fpga_runtime:stratix10:ppn=2 -d .`
  |GPU	                    |`qsub -I -l nodes=1:gpu:ppn=2 -d .`
  |CPU	                    |`qsub -I -l nodes=1:xeon:ppn=2 -d .`

>**Note**: For more information on how to specify compute nodes read, *[Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/)* in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **12 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI *[Intel® oneAPI Base Toolkit Get Started](https://devcloud.intel.com/oneapi/get_started/)* page.

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