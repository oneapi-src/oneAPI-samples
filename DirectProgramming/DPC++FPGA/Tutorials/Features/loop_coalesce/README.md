
# Coalescing Nested Loops
This FPGA tutorial demonstrates applying the `loop_coalesce` attribute to a nested loop in a task kernel to reduce the area overhead.

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. 

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               |  What the `loop_coalesce` attribute does <br> How `loop_coalesce` attribute affects resource usage and loop throughput <br> How to apply the `loop_coalesce` attribute to loops in your program <br> Which loops make good candidates for coalescing
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose
The `loop_coalesce` attribute enables you to direct the compiler to combine nested loops into a single loop. The attribute `[[intelfpga::loop_coalesce(N)]]` takes an integer argument `N`, that specifies how many nested loop levels that you want the compiler to attempt to coalesce.

**NOTE**: If you specify `[[intelfpga::loop_coalesce(1)]]` on nested loops, the compiler does not attempt to coalesce any of the nested loops.
### Example: Coalescing Two Loops 

```
[[intelfpga::loop_coalesce(2)]] 
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
Generally, coalescing loops can help reduce area usage by reducing the overhead needed for loop control. However, in some circumstances, coalescing loops can reduce kernel throughput. Scenarios where the `loop_coalesce` attribute can be applied to save area without a loss of throughput are those where: 

   1. The loops being coalesced have the same initiation interval (II).
   2. The exit condition computation for the resulting coalesced look is not complicated.

If the innermost coalesced loop has a very small trip count, `loop_coalesce` might actually improve throughput.


## Key Concepts
* Description of the `loop_coalesce` attribute 
* How `loop_coalesce` attribute affects resource usage and loop throughput 
* How to apply the `loop_coalesce` attribute to loops in your program 
* Determining which loops make good candidates for coalescing

## License  
This code sample is licensed under MIT license.


## Building the `loop_coalesce` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile or fpga_runtime) as well as whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/get-started/base-toolkit/](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:  
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
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
   * Compile for FPGA hardware (longer compile time, targets FPGA device): 
     ```
     make fpga
     ``` 
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_coalesce.fpga.tar.gz" download>here</a>.

### On a Windows* System
Note: `cmake` is not yet supported on Windows. A build.ninja file is provided instead. 

1. Enter the source file directory.
   ```
   cd src
   ```

2. Compile the design. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
      ```
      ninja fpga_emu
      ```

   * Generate the optimization report:

     ```
     ninja report
     ```
     If you are targeting Intel® PAC with Intel Stratix® 10 SX FPGA, instead use:
     ```
     ninja report_s10_pac
     ```     
   * Compiling for FPGA hardware is not yet supported on Windows.
 
 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `loop_coalesce_report.prj/reports/` or `loop_coalesce_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

On the main report page, scroll down to the section titled `Compile Estimated Kernel Resource Utilization Summary`. Each kernel name ends in the loop_coalesce attribute argument used for that kernel, e.g., KernelCompute<2> uses a loop_coalesce argument of 2. You can verify that the number of registers, MLABs and DSPs used for each kernel decreases after nested loops are coalesced.


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./loop_coalesce.fpga_emu     (Linux)
     loop_coalesce.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./loop_coalesce.fpga         (Linux)
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
     
