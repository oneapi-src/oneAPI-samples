
# Unrolling Loops
This FPGA tutorial demonstrates a simple example of unrolling loops to improve the throughput of a DPC++ FPGA program. 

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. 

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               |  Basics of loop unrolling <br> How to unroll loops in your program <br> Determining the optimal unroll factor for your program
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose

The loop unrolling mechanism is used to increase program parallelism by duplicating the compute logic within a loop. The number of times the loop logic is duplicated is called the *unroll factor*. Depending on whether the *unroll factor* is equal to the number of loop iterations or not, loop unroll methods can be categorized as *full-loop unrolling* and *partial-loop unrolling*.

### Example: Full-Loop Unrolling
```c++
// Before unrolling loop
#pragma unroll
for(i = 0 ; i < 5; i++){
  a[i] += 1;
}

// Equivalent code after unrolling
// There is no longer any loop 
a[0] += 1;
a[1] += 1;
a[2] += 1;
a[3] += 1;
a[4] += 1;
```
A full unroll is a special case where the unroll factor is equal to the number of loop iterations. Here, the the Intel® oneAPI DPC++ Compiler for FPGA instantiates five adders instead of the one adder.

### Example: Partial-Loop Unrolling

```c++
// Before unrolling loop
#pragma unroll 4
for(i = 0 ; i < 20; i++){
  a[i] += 1;
}

// Equivalent code after unrolling by a factor of 4
// The resulting loop has five (20 / 4) iterations
for(i = 0 ; i < 5; i++){
  a[i * 4] += 1;
  a[i * 4 + 1] += 1;
  a[i * 4 + 2] += 1;
  a[i * 4 + 3] += 1;
}
```
Each loop iteration in the "equivalent code" contains four unrolled invocations of the first. The Intel® oneAPI DPC++ Compiler (Beta) for FPGA instantiates four adders instead of one adder. Because there is no data dependency between iterations in the loop in this case, the compiler schedules all four adds in parallel.

### Determining the optimal unroll factor
In an FPGA design, unrolling loops is a common strategy to directly trade off on-chip resources for increased throughput. When selecting the unroll factor for specific loop, the intent is to improve throughput while minimizing resource utilization. It is also important to be mindful of other throughput constraints in your system, such as memory bandwidth.

### Tutorial design
This tutorial demonstrates this trade-off with a simple vector add kernel. The tutorial shows how increasing the unroll factor on a loop increases throughput... until another bottleneck is encountered. This example is constructed to run up against global memory bandwidth constraints.

The memory bandwidth on an Intel® Programmable Acceleration Card with Intel Arria® 10 GX FPGA system is about 6 GB/s. The tutorial design will likely run at around 300 MHz. In this design, the FPGA design processes a new iterations every cycle in a pipeline-parallel fashion. The theoretical computation limit for 1 adder is:

**GFlops**: 300 MHz \* 1 float = 0.3 GFlops

**Computation Bandwidth**: 300 MHz \* 1 float * 4 Bytes   = 1.2 GB/s

You repeat this back-of-the-envelope calculation for different unroll factors:

Unroll Factor  | GFlops (GB/s) | Compuation Bandwidth (GB/s)
------------- | ------------- | -----------------------
1   | 0.3 | 1.2
2   | 0.6 | 2.4
4   | 1.2 | 4.8
8   | 2.4 | 9.6
16  | 4.8 | 19.2

On an Intel® Programmable Acceleration Card with Intel Arria® 10 GX FPGA, it is reasonable to predict that this program will become memory-bandwidth limited when unroll factor grows from 4 to 8. Check this prediction by running the design following the instructions below.


## Key Concepts
* Basics of loop unrolling.
* How to unroll loops in your program.
* Determining the optimal unroll factor for your program.

## License  
This code sample is licensed under MIT license.


## Building the `loop_unroll` Tutorial

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
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/loop_unroll.fpga.tar.gz" download>here</a>.

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
Locate `report.html` in the `loop_unroll_report.prj/reports/` or `loop_unroll_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the Area Report and compare the FPGA resource utilization of the kernels with unroll factors of 1, 2, 4, 8, and 16. In particular, check the number of DSP resources consumed. You should see the area grow roughly linearly with the unroll factor.

You can also check the achieved system f<sub>MAX</sub> in order to verify the earlier calculations.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./loop_unroll.fpga_emu     (Linux)
     loop_unroll.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./loop_unroll.fpga         (Linux)
     ```

### Example of Output
```
Input Array Size:  67108864
UnrollFactor 1 kernel time : 255.749 ms
Throughput for kernel with UnrollFactor 1: 0.262 GFlops
UnrollFactor 2 kernel time : 140.285 ms
Throughput for kernel with UnrollFactor 2: 0.478 GFlops
UnrollFactor 4 kernel time : 68.296 ms
Throughput for kernel with UnrollFactor 4: 0.983 GFlops
UnrollFactor 8 kernel time : 44.567 ms
Throughput for kernel with UnrollFactor 8: 1.506 GFlops
UnrollFactor 16 kernel time : 39.175 ms
Throughput for kernel with UnrollFactor 16: 1.713 GFlops
PASSED: The results are correct
```

### Discussion of Results
The following table summarizes the execution time (in ms), throughput (in GFlops), and number of DSPs used for unroll factors of 1, 2, 4, 8, and 16 for a default input array size of 64M floats (2 ^ 26 floats) on Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA:

Unroll Factor  | Kernel Time (ms) | Throughput (GFlops) | Num of DSPs
------------- | ------------- | -----------------------| -------
1   | 242 | 0.277 | 1
2   | 127 | 0.528 | 2
4   | 63  | 1.065 | 4
8   | 46  | 1.459 | 8
16  | 44  | 1.525 | 16

Notice that when the unroll factor increases from 1 to 2 and from 2 to 4, the kernel execution time decreases by a factor of two. Correspondingly, the kernel throughput doubles. However, when the unroll factor is increase from 4 to 8 and from 8 to 16, the throughput does no longer scales by a factor of two at each step. The design is now bound by memory bandwidth limitations instead of compute unit limitations even though the hardware is replicated.

These performance differences will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.
