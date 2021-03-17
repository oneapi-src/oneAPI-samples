# Removing Loop Carried Dependencies
This tutorial demonstrates how to remove a loop-carried dependency to improve the performance of the FPGA device code.
 
***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.
 
| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | A technique to remove loop carried dependencies from your FPGA device code, and when to apply it
| Time to complete                  | 25 minutes
 


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
* Loop carried-dependencies and their impact on FPGA DPC++ kernel performance
* An optimization technique to break loop-carried data dependencies in critical loops

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
 
## Building the `loop_carried_dependency` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).
 
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
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:
 
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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_carried_dependency.fpga.tar.gz" download>here</a>.
 
### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:  
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
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
   * An FPGA hardware target is not provided on Windows*. 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.


### In Third-Party Integrated Development Environments (IDEs)
 
You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `loop_carried_dependency_report.prj/reports` or in `loop_carried_dependency_s10_pac_report.prj/reports` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the _Loops Analysis_ view of the report (under _Throughput Analysis_) and observe that the loop in block `UnOptKernel.B1` is showing _Serial exe: Data dependency_.  Click on the _source location_ field in the table to see the details for the loop. The maximum interleaving iterations of the loop is 1, as the loop is serialized.

Now, observe that the loop in block `OptKernel.B1` is not marked as _Serialized_.  The maximum Interleaving iterations of the loop is now 12.

## Running the Sample
 
 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./loop_carried_dependency.fpga_emu     (Linux)
     loop_carried_dependency.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./loop_carried_dependency.fpga         (Linux)
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


