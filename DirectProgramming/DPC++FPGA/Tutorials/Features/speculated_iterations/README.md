
# Speculated Iterations of a Loop
This FPGA tutorial demonstrates applying the `speculated_iterations` attribute to a loop in a task kernel to enable more efficient loop pipelining.

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. 

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               |  What the `speculated_iterations` attribute does <br> How to apply the `speculated_iterations` attribute to loops in your program <br> How to determine the optimal number of speculated iterations
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose
Loop speculation is an advanced loop pipelining optimization technique. It enables loop iterations to be initiated before determining whether they should have been initiated. "Speculated iterations" are those iterations that launch before the exit condition computation has completed. This is beneficial when the computation of the exit condition is preventing effective loop pipelining.

The `speculated_iterations` attribute is a loop attribute that enables you to directly control the number of speculated iterations for a loop.  The attribute  `[[intelfpga::speculated_iterations(N)]]` takes an integer argument `N` to specify the permissible number of iterations to speculate.

### Simple example
```
  [[intelfpga::speculated_iterations(1)]]
  while (sycl::log10(x) < N) {
      x += 1;
  }
  dst[0] = x;
```
The loop in this example will have one speculated iteration.
### Operations with side effects
When launching speculated iterations, operations with side-effects (such as stores to memory) must be predicated by the exit condition to ensure functional correctness. For this reason, operations with side-effects must be scheduled until after the exit condition has been computed.

### Optimizing the number of speculated iterations
Loop speculation is beneficial when the computation of the loop exit condition is the bottleneck preventing the compiler from achieving a smaller initiation interval (II). In such instances, increasing the number of speculated iterations often improves the II.  Note that this may also uncover additional bottlenecks preventing the further optimization of the loop.

However, adding speculated iterations is not without cost. They introduce overhead in nested loops, reducing overall loop occupancy. Consider the code snippet below:
```c++
for (size_t i = 0; i < kMany; ++i) {
  // The compiler may automatically infer speculated iterations 
  for (size_t j = 0; complex_exit_condition(j); ++j) {
    output[i,j] = some_function(input[i,j]);
  }
}
```
The *i+1*th  invocation of the inner loop cannot begin until all real and speculated iterations of its *i*th invocation have completed. This overhead is negligible if the number of speculated iterations is much less than the number of real iterations. However, when the inner loop's trip count is small on average, the overhead becomes non-negligible and the speculated iterations can become detrimental to throughput. In such circumstances, the `speculated_iterations` attribute can be used to *reduce* the number of speculated iterations chosen by the compiler's heuristics. 

In both increasing and decreasing cases, some experimentation is usually necessary. Choosing too new speculated iterations can increase the II because multiple cycles are required to evaluate the exit condition. Choosing too many speculated iterations creates unneeded "dead space" between sequential invocations of an inner loop.

### Tutorial example
In the tutorial design's kernel, the exit condition of the loop involves a logarithm and a compare operation. This complex exit condition prevents the loop from achieving ```II=1```. 

The design enqueues variants of the kernel with 0, 10 and 27 speculated iterations respectively to demonstrate the effect of the `speculated_iterations` attribute on the Intel® PAC with Intel Arria® 10 GX FPGA. Different numbers are chosen for the Intel® PAC with Intel Stratix® 10 SX FPGA accordingly.

## Key Concepts
* Description of the `speculated_iterations` attribute. 
* How to apply the `speculated_iterations` attribute to loops in your program.
* Optimizing the number of speculated iterations.

## License  
This code sample is licensed under MIT license.


## Building the `speculated_iterations` Tutorial

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
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/speculated_iterations.fpga.tar.gz" download>here</a>.

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
Locate `report.html` in the `speculated_iterations_report.prj/reports/` or `speculated_iterations_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

In the "Loop Analysis" section of the report, check the II of the loop in each version of the kernel. Use the kernel with 0 speculated iteration as a base version, check its loop II as a hint for the ideal number for speculated iterations. The information shown below is from compiling on the Intel® PAC with Intel Arria® 10 GX FPGA.

* When the number of  `speculated iterations` is set to 0, the loop II is 27.
* Setting the `speculated iterations` to 27 yielded an II of 1.
* Setting the `speculated iterations` to an intermediate value of 10 results in an II of 3. 


These results make sense when you recall that the loop exit computation has a latency of 27 cycles (suggested by looking at the loop II with 0 speculation). With no speculation, a new iteration can only be launched every 27 cycles. Increasing the speculation to 27 enables a new iteration to launch every cycle. Reducing the speculation to 10 results in an II of 3 because 10 speculated iteration multipled by 3 cycles between iterations leaves 30 cycles in which to compute the exit condition, sufficient to cover the 27-cycle exit condition. 

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./speculated iterations.fpga_emu     (Linux)
     speculated iterations.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./speculated iterations.fpga         (Linux)
     ```

### Example of Output
```
Speculated Iterations: 0 -- kernel time: 8564.98 ms
Performance for kernel with 0 speculated iterations: 11675 MFLOPs
Speculated Iterations: 10 -- kernel time: 952 ms
Performance for kernel with 10 speculated iterations: 105076 MFLOPs
Speculated Iterations: 27 -- kernel time: 317 ms
Performance for kernel with 27 speculated iterations: 315181 MFLOPs
PASSED: The results are correct
```
The execution time and throughput for each kernel is displayed. 

Note that this performance difference will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.
     
