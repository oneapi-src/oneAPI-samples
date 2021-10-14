
# Loop Fusion
This FPGA tutorial demonstrates how loop fusion is used and affects performance.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               |  Basics of loop-carried dependencies <br> The notion of a loop-carried dependence distance <br> What constitutes a *safe* dependence distance <br> How to aid the compiler's dependence analysis to maximize performance
| Time to complete                  | 30 minutes



## Purpose
In order to understand and apply loop fusion to loops in your design, it is necessary to understand the motivation and consequences of loop fusion. Unlike many attributes that can improve a design's performance, loop fusion may have functional implications. Using it incorrectly may result in undefined behavior for your design!

### Loop Fusion
Loop fusion is a compiler transformation in which adjacent loops are merged into a single loop over the same index range. This transformation is typically applied to reduce loop overhead and improve run-time performance. Loop control structures represent a significant area overhead on designs produced by the Intel® oneAPI DPC++ compiler. Fusing two loops into one loop reduces the number of required loop-control structures, which reduces overhead.
 
In addition, fusing outer loops can introduce concurrency where there was previously none. Combining the bodies of two adjacent loops L<sub>j</sub> and L<sub>k</sub> forms a single loop L<sub>f</sub> with a body that spans the bodies of L<sub>j</sub> and L<sub>k</sub>. The combined loop body creates an opportunity for operations that are independent across a given iteration of L<sub>j</sub> and L<sub>k</sub> to execute concurrently. In effect, the two loops now execute as one, in a lockstep fashion, giving latency improvements.

The Intel® oneAPI DPC++ compiler attempts to fuse adjacent loops by default when profitable and memory dependencies allow. Compiler loop fusion heuristics can be overridden using the `fpga_loop_fuse<N>(f)` function. 

The `fpga_loop_fuse<N>(f)` function takes a function `f` containing loops, and an optional template parameter `N`, which specifies the number of nesting depths in which fusion should be performed.  For example, consider a function `f` containing the following loops:
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
When `N=1`, `fpga_loop_fuse<N>(f)` fuses L<sub>11</sub> with L<sub>21</sub>, but not L<sub>12</sub> with L<sub>22</sub>. When `N=2`, `fpga_loop_fuse<N>(f)` fuses L<sub>11</sub> with L<sub>21</sub>, and fuses L<sub>12</sub> with L<sub>22</sub>. 

### Negative-Distance Dependencies

The case when there are two adjacent loops L<sub>j</sub> and L<sub>k</sub>, and iteration *m* of L<sub>k</sub> depends on iteration *n* > *m* of L<sub>j</sub> is known as a *negative-distance dependency*. The Intel® oneAPI DPC++ compiler will not fuse loops that are believed to have a negative-distance dependency, even when the `fpga_loop_fuse<N>(f)` function is used. 

#### Overriding Compiler Memory Checks

The compiler can be told to ignore memory safety checks by using the `fpga_loop_fuse_independent<N>(f)` function. This function requires the same parameters as the `fpga_loop_fuse<N>(f)` function. 

***IMPORTANT***: Functional incorrectness may result if `fpga_loop_fuse_independent<N>(f)` is applied where a negative-distance dependency exists.

### Turning Off Loop Fusion

The `intel::nofusion` attribute is applied to a loop to tell the compiler that the loop should not be fused with any other.

### Testing the Tutorial
The file`loop_fusion.cpp` contains three kernels: 

|Kernel Name|Description  |
|--|--|
|`DefaultFusionKernel`| This kernel contains two loops with equal trip counts which fuse by default. |
|`NoFusionKernel`| This kernel has two loops with equal trip counts as in `DefaultFusionKernel`, but the compiler is told not to fuse the loops using the `intel::nofusion` attribute.  |
|`DefaultNoFusionKernel`| This kernel contains two loops with unequal trip counts, which the compiler does not fuse by default.   |
|`FusionFunctionKernel`| This kernel contains two loops with unequal trip counts as in `DefaultNoFusionKernel`, but the compiler is told to fuse the loops using the `fpga_loop_fuse<N>(f)` function. |

## Key Concepts
* Basics of loop fusion
* The reasons for loop fusion
* Understanding negative-distance dependencies
* How to use loop fusion to increase performance

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `loop_ivdep` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

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
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_ivdep.fpga.tar.gz" download>here</a>.

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
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=<board-support-package>:<board-variant>
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
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ``` 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.
 
 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `loop_fusion_report.prj/reports/` or `loop_fusion_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the Loops Analysis section of the optimization report and notice that two loops were fused to one in both `DefaultFusionKernel` and in `FusionFunctionKernel`, but not in `NoFusionKernel` or in `DefaultNoFusionKernel`.

Navigate to the Area Analysis of the System under Area Analysis. The Kernel System section displays the area consumption of each kernel. Notice the area savings when loop fusion is on by default in `DefaultFusionKernel`, against when it is manually turned off in `NoFusionKernel`. 

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./loop_fusion.fpga_emu     (Linux)
     loop_fusion.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./loop_fusion.fpga         (Linux)
     ```

### Example of Output

```
SAFELEN: 1 -- kernel time : 50.9517 ms
Throughput for kernel with SAFELEN 1: 1286KB/s
SAFELEN: 128 -- kernel time : 10 ms
Throughput for kernel with SAFELEN 128: 6277KB/s
PASSED: The results are correct
```

### Discussion of Results

The following table summarizes the execution time (in ms) and throughput (in MFlops) for `safelen` parameters of 1 (redundant attribute) and 128 (`kRowLength`) for a default input matrix size of 128 x 128 floats on Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA and the Intel® oneAPI DPC++ Compiler.

Safelen | Kernel Time (ms) | Throughput (KB/s)
------------- | ------------- | -----------------------
1     | 50 | 1320
128   | 10 | 6403

With the `ivdep` attribute applied with the maximum safe `safelen` parameter, the kernel execution time is decreased by a factor of ~5. 

Note that this performance difference will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.
