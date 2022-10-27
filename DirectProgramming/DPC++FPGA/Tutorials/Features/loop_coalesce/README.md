
# Coalescing Nested Loops
This FPGA tutorial demonstrates applying the `loop_coalesce` attribute to a nested loop in a task kernel to reduce the area overhead.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg; 10 GX FPGA <br> Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX) <br> Intel&reg; FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel&reg; FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler <br> Intel&reg; FPGA Add-On for oneAPI Base Toolkit
| What you will learn               |  What the `loop_coalesce` attribute does <br> How `loop_coalesce` attribute affects resource usage and loop throughput <br> How to apply the `loop_coalesce` attribute to loops in your program <br> Which loops make good candidates for coalescing
| Time to complete                  | 15 minutes



## Purpose
The `loop_coalesce` attribute enables you to direct the compiler to combine nested loops into a single loop. The attribute `[[intel::loop_coalesce(N)]]` takes an integer argument `N`, that specifies how many nested loop levels that you want the compiler to attempt to coalesce.

**NOTE**: If you specify `[[intel::loop_coalesce(1)]]` on nested loops, the compiler does not attempt to coalesce any of the nested loops.
### Example: Coalescing Two Loops

```
[[intel::loop_coalesce(2)]]
for (int i = 0; i < N; i++)
  for (int j = 0; j < M; j++)
    sum[i][j] += i+j;
```
The compiler coalesces the two loops together so that they execute as if they were a single loop written as follows:

```
int i = 0;
int j = 0;
while(i < N){
  sum[i][j] += i+j;
  j++;
  if (j == M){
    j = 0;
    i++;
  }
}
```

### Identifying Which Loops to Coalesce
Generally, coalescing loops can help reduce area usage by reducing the overhead needed for loop control. However, in some circumstances, coalescing loops can reduce kernel throughput. Scenarios where the `loop_coalesce` attribute can be applied to save area without a loss of throughput, are those where:

   1. The loops being coalesced have the same initiation interval (II).
   2. The exit condition computation for the resulting coalesced look is not complicated.

If the innermost coalesced loop has a very small trip count, `loop_coalesce` might actually improve throughput.

### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.

## Key Concepts
* Description of the `loop_coalesce` attribute.
* How `loop_coalesce` attribute affects resource usage and loop throughput.
* How to apply the `loop_coalesce` attribute to loops in your program.
* Determining which loops make good candidates for coalescing.


## Building the `loop_coalesce` Tutorial

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see **Use the setvars Script** for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel&reg; oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel&reg; oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


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
   * Compile for simulation (fast compile time, targets simulated FPGA device):
     ```
     make fpga_sim
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_coalesce.fpga.tar.gz" download>here</a>.

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
   * Compile for simulation (fast compile time, targets simulated FPGA device):
     ```
     nmake fpga_sim
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```


> **Note**: The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Troubleshooting
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).

## Examining the Reports
Locate `report.html` in the `loop_coalesce_report.prj/reports/` directory. Open the report in Chrome*, Firefox*, Edge*, or Internet Explorer*.

On the main report page, scroll down to the section titled `Compile Estimated Kernel Resource Utilization Summary`. Each kernel name ends in the loop_coalesce attribute argument used for that kernel, e.g., KernelCompute<2> uses a loop_coalesce argument of 2. You can verify that the number of registers, MLABs and DSPs used for each kernel decreases after nested loops are coalesced.


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```bash
     ./loop_coalesce.fpga_emu     (Linux)
     loop_coalesce.fpga_emu.exe   (Windows)
     ```
 2. Run the sample on the FPGA simulator device:
     ```bash
     ./loop_coalesce.fpga_sim     (Linux)
     loop_coalesce.fpga_sim.exe   (Windows)
     ```
 3. Run the sample on the FPGA device:
     ```bash
     ./loop_coalesce.fpga         (Linux)
     loop_coalesce.fpga.exe       (Windows)
     ```

### Example of Output

```
Loop Coalesce: 1 -- kernel time : 156 microseconds
Throughput for kernel with coalesce_factor 1: 6550KB/S
Loop Coalesce: 2 -- kernel time : 113 microseconds
Throughput for kernel with coalesce_factor 2: 9064KB/S
PASSED: The results are correct

```

### Discussion of Results
The execution time and throughput for each kernel is displayed. Applying the `loop_coalesce` attribute in this example reduced the kernel execution time by a factor of ~1.5. Note that you will only see this result when executing on FPGA hardware. The emulator will generally not reflect performance differences.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
