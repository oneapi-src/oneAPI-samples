# Private Copies
This FPGA tutorial explains how to use the `private_copies` attribute to improve concurrent outer loop iterations.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | The basic usage of the `private_copies` attribute <br> How the `private_copies` attribute affects loop throughput and resource use <br> How to apply the `private_copies` attribute in your program <br> How to identify the correct `private_copies` factor for your program
| Time to complete                  | 15 minutes



## Purpose
This tutorial demonstrates a simple example of applying the `private_copies` attribute to an array within a loop in a task kernel to trade off the on-chip memory use and throughput of the loop.

### Description of the `private_copies` Attribute
The `private_copies` attribute is a memory attribute that enables you to control the number of private copies of any variable or array declared inside a pipelined loop. These private copies allow multiple iterations of the loop to run concurrently by providing them their own private workspaces. The number of concurrent loop iterations is limited by the number of private copies specified by the `private_copies` attribute.

#### Example: 

Kernels in this tutorial design apply `[[intel::private_copies(N)]]` to an array declared within an outer loop that is used by subsequent inner loops. These inner loops perform a global memory access before storing the results. The following is an example of such a loop:

```cpp
[[intel::private_copies(2)]]
for (size_t i = 0; i < kMaxIter; i++) {
  [[intel::private_copies(2)]] int a[kSize];
  for (size_t j = 0; j < kSize; j++) {
    a[j] = accessor_array[(i * 4 + j) % kSize] * shift;
  }
  for (size_t j = 0; j < kSize; j++)
    r += a[j];
}    
```

In this example, you only need to have two private copies of array `a` in order to have 2 concurrent outer loop iterations. The `private_copies` attribute in this example forces the compiler to create two private copies of the array `a`. In general, passing the parameter `N` to the `private_copies` attribute limits the number of private copies created for array `a` to `N`, which in turn limits the concurrency of the outer loop to `N`.

### Identifying the Correct `private_copies` Factor
Generally, increasing the number of private copies of an array within a loop situated in a task kernel will increase the throughput of that loop at the cost of increased memory usage. However, in most cases, there is a limit beyond which increasing the number of private copies does not have any further effect on the throughput of the loop. That limit is the maximum exploitable concurrency of the outer loop. 

The correct `private_copies` factor for a given array depends on your goals for the design, the criticality of the loop in question, and its impact on your design's overall throughput. A typical design flow may be to: 
1. Experiment with different values of `private_copies`. 
2. Observe what impact the values have on the overall throughput and memory usage of your design.
3. Choose the appropriate value that allows you to achieve your desired throughput and area goals.

## Key Concepts
* The basic usage of the `private_copies` attribute 
* How the `private_copies` attribute affects loop throughput and resource usage
* How to apply the `private_copies` attribute to variables or arrays in your program
* How to identify the correct `private_copies` factor for your program

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `private_copies` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

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
3. (Optional) As the FPGA hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/private_copies.fpga.tar.gz" download>here</a>.

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
Locate `report.html` in the `private_copies_report.prj/reports/` or `private_copies_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

On the main report page, scroll down to the section titled "Estimated Resource Usage". Each kernel name ends in the `private_copies` attribute argument used for that kernel, e.g., `kernelCompute1` uses a `private_copies` attribute value of 1. You can verify that the number of RAMs used for each kernel increases with the private_copies value used, with the exception of private_copies 0, which instructs the compiler to choose a default value.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./private_copies.fpga_emu     (Linux)
     private_copies.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./private_copies.fpga         (Linux)
     ```


### Example of Output
```
PASSED_fpga_compile
Num private_copies 0 kernel time : 1441.45 ms
Throughput for kernel with private_copies 0: 0.568 GFlops
Num private_copies 1 kernel time : 2916.440 ms
Throughput for kernel with private_copies 1: 0.281 GFlops
Num private_copies 2 kernel time : 1458.260 ms
Throughput for kernel with private_copies 2: 0.562 GFlops
Num private_copies 3 kernel time : 1441.450 ms
Throughput for kernel with private_copies 3: 0.568 GFlops
Num private_copies 4 kernel time : 1441.448 ms
Throughput for kernel with private_copies 4: 0.568 GFlops
PASSED: The results are correct
```

### Discussion of Results

The stdout output shows the throughput (GFlops) for each kernel. 

When run on the Intel® PAC with Intel Arria10® 10 GX FPGA hardware board, we see that the throughput doubles from using private_copies 1 to private_copies 2. Further increasing the value of private_copies does not increase the throughput achieved, i.e., increasing the private_copies above 2 will spend additional RAM resources for no additional throughput gain. As such, for this tutorial design, maximal throughput is best achieved by using private_copies 2. 

Using private_copies 0 (or equivalently omitting the attribute entirely) also produced good throughput, indicating that the compiler's default heuristic chose a private_copies value of 2 or higher in this case.

When run on the FPGA emulator, the private_copies attribute has no effect on runtime. You may notice that the emulator achieved higher throughput than the FPGA in this example. This is because this trivial example uses only a tiny fraction of the spatial compute resources available on the FPGA.
