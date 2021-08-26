# Caching On-Chip Memory to Improve Loop Performance
This FPGA tutorial demonstrates how to build a simple cache (implemented in FPGA registers) to store recently-accessed memory locations so that the compiler can achieve II=1 on critical loops in task kernels.


***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | How and when to implement the on-chip memory cache optimization
| Time to complete                  | 30 minutes



## Purpose
In DPC++ task kernels for FPGA, our objective is to achieve an initiation interval (II) of 1 on performance-critical loops. This means that a new loop iteration is launched on every clock cycle, maximizing the loop's throughput. 

When the loop contains a loop-carried variable implemented in on-chip memory, the compiler often *cannot* achieve II=1 because the memory access takes more than one clock cycle. If the updated memory location may be needed on the next loop iteration, the next iteration must be delayed to allow time for the update, hence II > 1.

The on-chip memory cache technique breaks this dependency by storing recently-accessed values in a cache capable of a 1-cycle read-modify-write operation. The cache is implemented in FPGA registers rather than on-chip memory. By pulling memory accesses preferentially from the register cache, the loop-carried dependency is broken.

### When is the on-chip memory cache technique applicable?

***Failure to achieve II=1 because of a loop-carried memory dependency in on-chip memory***:
The on-chip memory cache technique is applicable if the compiler could not pipeline a loop with II=1 because of an on-chip memory dependency. (If the compiler could not achieve II=1 because of a *global* memory dependency, this technique does not apply as the access latencies are too great.)

To check this for a given design, view the "Loops Analysis" section of its optimization report. The report lists the II of all loops and explains why a lower II is not achievable. Check whether the reason given resembles "the compiler failed to schedule this loop with smaller II due to memory dependency". The report will describe the "most critical loop feedback path during scheduling". Check whether this includes on-chip memory load/store operations on the critical path.

***An II=1 loop with a load operation of latency 1***:
The compiler is capable of reducing the latency of on-chip memory accesses to achieve II=1. In doing so, the compiler makes a trade-off by sacrificing f<sub>MAX</sub> to improve the II. 

In a design with II=1 critical loops but lower than desired f<sub>MAX</sub>, the on-chip memory cache technique may still be applicable. It can help recover f<sub>MAX</sub> by enabling the compiler to achieve II=1 with a higher latency memory access.

To check whether this is the case for a given design, view the "Kernel Memory Viewer" section of the optimization report. Select the on-chip memory of interest from the Kernel Memory List, and mouse over the load operation "LD" to check its latency. If the latency of the load operation is 1, this is a clear sign that the compiler has attempted to sacrifice f<sub>MAX</sub> to improve loop II.


### Implementing the on-chip memory cache technique

The tutorial demonstrates the technique using a program that computes a histogram. The histogram operation accepts an input vector of values, separates the values into buckets, and counts the number of values per bucket. For each input value, an output bucket location is determined, and the count for the bucket is incremented. This count is stored in the on-chip memory, and the increment operation requires reading from memory, performing the increment, and storing the result. This read-modify-write operation is the critical path that can result in II > 1.

To reduce II, the idea is to store recently-accessed values in an FPGA register-implemented cache that is capable of a 1-cycle read-modify-write operation. If the memory location required on a given iteration exists in the cache, it is pulled from there. The updated count is written back to *both* the cache and the on-chip memory. The `ivdep` pragma is added to inform the compiler that if a loop-carried variable (namely, the variable storing the histogram output) is needed within `CACHE_DEPTH` iterations, it is guaranteed to be available right away.

### Selecting the cache depth

While any value of `CACHE_DEPTH` results in functional hardware, the ideal value of `CACHE_DEPTH` requires some experimentation. The depth of the cache needs to roughly cover the latency of the on-chip memory access. To determine the correct value, it is suggested to start with a value of 2 and then increase it until both II = 1 and load latency > 1. In this tutorial, a `CACHE_DEPTH` of 5 is needed. 

Each iteration takes only a few moments by running `make report` (refer to the section below on how to build the design). It is important to find the *minimal* value of `CACHE_DEPTH` that results in a maximal performance increase. Unnecessarily large values of `CACHE_DEPTH` consume unnecessary FPGA resources and can reduce f<sub>MAX</sub>. Therefore, at a `CACHE_DEPTH` that results in II=1 and load latency = 1, if further increases to `CACHE_DEPTH` show no improvement, `CACHE_DEPTH` should not be increased any further.

Two versions of the histogram kernel are implemented in the tutorial: one with and one without caching. The report shows II > 1 for the loop in the kernel without caching and II = 1 for the one with caching.

## Key Concepts
* How to implement the on-chip memory cache optimization technique
* The scenarios in which this technique benefits performance 
* How to tune the cache depth

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `onchip_memory_cache` Tutorial

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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/onchip_memory_cache.fpga.tar.gz" download>here</a>.

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
Locate `report.html` in the `onchip_memory_cache_report.prj/reports/` or `onchip_memory_cache_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Compare the Loop Analysis reports with and without the on-chip memory cache optimization, as described in the "When is the on-chip memory cache technique applicable?" section.


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./onchip_memory_cache.fpga_emu     (Linux)
     onchip_memory_cache.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./onchip_memory_cache.fpga         (Linux)
     ```

### Example of Output

```
Platform name: Intel(R) FPGA SDK for OpenCL(TM)
Device name: pac_a10 : Intel PAC Platform (pac_ee00000)


Number of inputs: 16777216
Number of outputs: 64

Beginning run without local memory caching.

Verification PASSED

Kernel execution time: 0.114106 seconds
Kernel throughput without caching: 560.884047 MB/s

Beginning run with local memory caching.

Verification PASSED

Kernel execution time: 0.059061 seconds
Kernel throughput with caching: 1083.623184 MB/s
```

### Discussion of Results

A test compile of this tutorial design achieved an f<sub>MAX</sub> of approximately 250 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results are shown in the following table:

Configuration | Execution Time (ms) | Throughput (MB/s)
-|-|-
Without caching | 0.153 | 418
With caching | 0.08 | 809

When caching is used, performance notably increases. As previously mentioned, this technique should result in an II reduction, which should lead to a throughput improvement. The technique can also improve f<sub>MAX</sub> if the compiler had previously implemented a latency=1 load operation, in which case the f<sub>MAX</sub> increase should result in a further throughput improvement.

