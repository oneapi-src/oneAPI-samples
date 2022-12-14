# Removing Loop Carried Dependencies
This tutorial demonstrates how to remove a loop-carried dependency to improve the performance of the FPGA device code.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg; 10 GX FPGA <br> Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX) <br> Intel&reg; FPGA 3rd party / custom platforms with oneAPI support <br> **Note**: Intel&reg; FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | A technique to remove loop carried dependencies from your FPGA device code, and when to apply it.
| Time to complete                  | 25 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstatres a design pattern.

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

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/DPC++FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/DPC++FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/DPC++FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/DPC++FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/DPC++FPGA/README.md#documentation), etc.

## Purpose
This tutorial demonstrates how to remove a loop-carried dependency in FPGA device code. A snippet of the baseline unoptimized code (the `Unoptimized` function in `src/loop_carried_dependency.cpp`) is given below:

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

In the unoptimized kernel, a sum is computed over two loops.  The inner loop sums over the `a` data and the outer loop over the `b` data. Since the value `sum` is updated in both loops, this introduces a _loop carried dependency_ that causes the outer loop to be serialized, allowing only one invocation of the outer loop to be active at a time, which reduces performance.

A snippet of the optimized code (the `Optimized` function in `src/loop_carried_dependency.cpp`) is given below, which removes the loop carried dependency on the `sum` variable:

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
Look at the _Compiler Report > Throughput Analysis > Loop Analysis_ section in the reports. The report lists the II and details for each loop. The technique presented in this tutorial may be applicable if the _Brief Info_ of the loop shows _Serial exe: Data dependency_.  The details pane may provide more information:
```
* Iteration executed serially across _function.block_. Only a single loop iteration will execute inside this region due to data dependency on variable(s):
    * sum (_filename:line_)
```

## Key Concepts
- Loop carried-dependencies and their impact on FPGA SYCL kernel performance
- An optimization technique to break loop-carried data dependencies in critical loops

## Building the `loop_carried_dependency` Tutorial

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

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     make fpga_emu
     ```
   * Generate the optimization report:
     ```
     make report
     ```
   * Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
     ```
     make fpga_sim
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_carried_dependency.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report:
     ```
     nmake report
     ```
   * Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
     ```
     nmake fpga_sim
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

*Note:* The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Examining the Reports
Locate `report.html` in the `loop_carried_dependency_report.prj/reports` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the _Loops Analysis_ view of the report (under _Throughput Analysis_) and observe that the loop in block `UnOptKernel.B1` is showing _Serial exe: Data dependency_.  Click on the _source location_ field in the table to see the details for the loop. The maximum interleaving iterations of the loop is 1, as the loop is serialized.

Now, observe that the loop in block `OptKernel.B1` is not marked as _Serialized_.  The maximum Interleaving iterations of the loop is now 12.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./loop_carried_dependency.fpga_emu     (Linux)
     loop_carried_dependency.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA simulator device:
     ```
     ./loop_carried_dependency.fpga_sim     (Linux)
     loop_carried_dependency.fpga_sim.exe   (Windows)
     ```
3. Run the sample on the FPGA device:
     ```
     ./loop_carried_dependency.fpga         (Linux)
     loop_carried_dependency.fpga.exe       (Windows)
     ```

### Example of Output
```
Number of elements: 16000
Run: Unoptimized:
kernel time : 10685.3 ms
Run: Optimized:
kernel time : 2736.47 ms
PASSED
```
### Discussion of Results

In the tutorial example, applying the optimization yields a total execution time reduction by almost a factor of 4.  The Initiation Interval (II) for the inner loop is 12 because a double floating point add takes 11 cycles on the FPGA.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
