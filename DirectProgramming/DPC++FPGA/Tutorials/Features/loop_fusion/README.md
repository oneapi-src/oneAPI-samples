

# Loop Fusion
This FPGA tutorial demonstrates how loop fusion is used and how it affects performance.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               |  Basics of loop fusion<br/>The reasons for loop fusion<br/>How to use loop fusion to increase performance<br/>Understanding safe application of loop fusion
| Time to complete                  | 20 minutes



## Purpose
In order to understand and apply loop fusion to loops in your design, it is necessary to understand the motivation and consequences of loop fusion. 

### Loop Fusion
Loop fusion is a compiler transformation in which adjacent loops are merged into a single loop over the same index range. This transformation is typically applied to reduce loop overhead and improve runtime performance. Loop control structures can represent a significant area overhead on designs produced by the Intel® oneAPI DPC++ compiler. Fusing two loops into one loop reduces the number of required loop-control structures, which reduces overhead.
 
In addition, fusing outer loops can introduce concurrency where there was previously none. Consider two adjacent loops L<sub>j</sub> and L<sub>k</sub>. Within each loop, independent operations can be run concurrently, but concurrency cannot be attained <i>across</i> the loops. Combining the bodies of L<sub>j</sub> and L<sub>k</sub> forms a single loop L<sub>f</sub> with a body that spans the bodies of L<sub>j</sub> and L<sub>k</sub>. In the combined loops, concurrency can be attained for independent instructions which were formerly in separate loops. In effect, the two loops now execute as one in L<sub>f</sub> in a lockstep fashion, providing possible latency improvements.

Loop fusion joins loops at the same nesting level. The merging of nested loops is known as *loop coalescing*, and tools to achieve this are described in the documentation and in the [`loop_coalesce` code sample](https://github.com/oneapi-src/oneAPI-samples/tree/da084668be646bfe9f788da7530a3efb3494e8c7/DirectProgramming/DPC++FPGA/Tutorials/Features/loop_coalesce).

#### Default Loop Fusion

The Intel® oneAPI DPC++ compiler attempts to fuse adjacent loops by default when profitable and when memory dependencies allow. For example, the compiler will not fuse loops by default when two adjacent loops have unequal trip counts, if only one of the two loops has stall-free logic, or if only one of the two loops is tagged with the `intel::ivdep` attribute. The `intel::nofusion` attribute should be applied to a loop to tell the compiler not to fuse that loop with others.

#### Explicit Loop Fusion

The case when there are two adjacent loops L<sub>j</sub> and L<sub>k</sub>, and iteration *m* of L<sub>k</sub> depends on iteration *n* > *m* of L<sub>j</sub> is known as a *negative-distance dependency*.  A negative-distance dependency cannot be fulfilled when loop fusion is performed. The Intel® oneAPI DPC++ compiler will therefore not fuse loops that are believed to have a negative-distance dependency, even when the `fpga_loop_fuse<N>(f)` function is used. 

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

The compiler may conservatively not fuse a pair of loops due to a suspected memory dependency when such a dependency may not exist. In this situation the compiler can be told to ignore memory safety checks by using the `fpga_loop_fuse_independent<N>(f)` function. This function requires the same parameters as the `fpga_loop_fuse<N>(f)` function. 

***IMPORTANT***: Functional incorrectness may result if `fpga_loop_fuse_independent<N>(f)` is applied where a negative-distance dependency exists.

### Testing the Tutorial
The file `loop_fusion.cpp` contains four kernels, all of which contain an outer loop and two inner loops.

|Kernel Name|Description  |
|--|--|
|`DefaultFusionKernel`| This kernel contains two inner loops with equal trip counts which fuse by default. |
|`NoFusionKernel`| This kernel has two inner loops with equal trip counts as in `DefaultFusionKernel`, but the compiler is instructed not to fuse the loops using the `intel::nofusion` attribute.  |
|`DefaultNoFusionKernel`| This kernel contains two inner loops with unequal trip counts, which the compiler does not fuse by default.   |
|`FusionFunctionKernel`| This kernel contains two inner loops with unequal trip counts as in `DefaultNoFusionKernel`, but the compiler is instructed to fuse the loops using the `fpga_loop_fuse<N>(f)` function. |

## Key Concepts
* Basics of loop fusion
* The reasons for loop fusion
* How to use loop fusion to increase performance
* Understanding safe application of loop fusion

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the Loop Fusion Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_fusion.fpga.tar.gz" download>here</a>.

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

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `loop_fusion_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the Loops Analysis section of the optimization report under Throughput Analysis and notice that two loops were fused to one in both `DefaultFusionKernel` and in `FusionFunctionKernel`, but not in `NoFusionKernel` or in `DefaultNoFusionKernel`.

In both cases where fusion has occurred, the number of loop cycles has decreased, since the total number of loop iterations decreased due to loop fusion, while the II, speculated iterations and latency are the same in the fused and non-fused loops. 

Navigate to the Area Analysis of the system under Area Analysis. The Kernel System section displays the area consumption of each kernel. Notice the area savings when loop fusion is performed in`DefaultFusionKernel`, against when it is off in `NoFusionKernel`.  As well, notice the area savings when loop fusion is manually turned on in`FusionFunctionKernel`, against when it is off by default in `DefaultNoFusionKernel`.


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
Throughput for kernel with default loop fusion and with equally-sized loops: 1.48999 Ops/ns
Throughput for kernel with the nofusion attribute and with equally-sized loops: 0.745144 Ops/ns
Throughput for kernel without fusion by default with unequally-sized loops: 0.745192 Ops/ns
Throughput for kernel with a loop fusion function with unequally-sized loops: 1.49017 Ops/ns
PASSED: The results are correct
```

### Discussion of Results

Loop fusion increases the throughput by ~100% in both the cases with equally-sized and unequally-sized loops. 

Note that this performance difference will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.
