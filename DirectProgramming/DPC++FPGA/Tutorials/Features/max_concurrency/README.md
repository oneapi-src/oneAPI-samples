# Maximum Concurrency of a Loop
This FPGA tutorial explains how to use the max_concurrency attribute for loops.

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. 

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | The basic usage of the `max_concurrency` attribute <br> How the `max_concurrency` attribute affects loop throughput and resource use <br> How to apply the `max_concurrency` attribute to loops in your program <br> How to identify the correct `max_concurrency` factor for your program
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates a simple example of applying the `max_concurrency` attribute to a loop in a task kernel to trade off the on-chip memory use and throughput of the loop.

### Description of the `max_concurrency` Attribute
The `max_concurrency` attribute is a loop attribute that enables you to control the number of simultaneously executed loop iterations. To enable this simultaneous execution, the compiler creates copies of any memory that is private to a single iteration. These copies are called private copies. The greater the permitted concurrency, the more private copies the compiler must create. 

#### Example: 

Kernels in this tutorial design apply `[[intelfpga::max_concurrency(N)]]` to an outer loop that contains two inner loops, which perform a partial sum computation on an input array, storing the results in a private (to the outer loop) array `a1`. The following is an example of a loop nest:

```
[[intelfpga::max_concurrency(1)]]
for (size_t i = 0; i < max_iter; i++) {                                                      
  float a1[size];                                                                              
  for (int j = 0; j < size; j++)                                                               
    a1[j] = accessorA[i * 4 + j] * shift;                                                      
  for (int j = 0; j < size; j++)                                                               
    result += a1[j];                                                                           
}   
```

In this example, the maximum concurrency allowed for the outer loop is 1, that is, only one iteration of the outer loop is allowed to be simultaneously executing at any given moment. The `max_concurrency` attribute in this example forces the compiler to create exactly one private copy of the array `a1`. Passing the parameter `N` to the `max_concurrency` attribute limits the concurrency of the loop to `N` simultaneous iterations, and `N` private copies of privately-declared arrays in that loop.

### Identifying the Correct `max_concurrency` Factor
Generally, increasing the maximum concurrency allowed for a loop through the use of the `max_concurrency` attribute increases the throughput of that loop at the cost of increased memory resource use. Additionally, in nearly all cases, there is a point at which increasing the maximum concurrency does not have any further effect on the throughput of the loop, as the maximum exploitable concurrency of that loop has been achieved. 

The correct `max_concurrency` factor for a loop depends on the goals of your design, the criticality of the loop in question, and its impact on the overall throughput of your design. A typical design flow may be to: 
1. Experiment with different values of `max_concurrency`. 
2. Observe what impact the values have on the overall throughput and memory use of your design.
3. Choose the appropriate value that allows you to achive your desired throughput and area goals.

## Key Concepts
* The basic usage of the `max_concurrency` attribute 
* How the `max_concurrency` attribute affects loop throughput and resource use
* How to apply the `max_concurrency` attribute to loops in your program
* How to identify the correct `max_concurrency` factor for your program

## License  
This code sample is licensed under MIT license.

## Building the `max_concurrency` Tutorial

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
3. (Optional) As the FPGA hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/max_concurrency.fpga.tar.gz" download>here</a>.

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
Locate `report.html` in the `max_concurrency_report.prj/reports/` or `max_concurrency_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

On the main report page, scroll down to the section titled "Estimated Resource Usage". Each kernel name ends in the max_concurrency attribute argument used for that kernel, e.g., `kernelCompute1` uses a max_concurrency attribute value of 1. You can verify that the number of RAMs used for each kernel increases with the max_concurrency value used, with the exception of max_concurrency 0, which instructs the compiler to choose a default value.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./max_concurrency.fpga_emu     (Linux)
     max_concurrency.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./max_concurrency.fpga         (Linux)
     ```


### Example of Output
```
Max concurrency 0 kernel time : 1459.89 ms
Throughput for kernel with max_concurrency 0: 0.561 GFlops
Max concurrency 1 kernel time : 2890.810 ms
Throughput for kernel with max_concurrency 1: 0.283 GFlops
Max concurrency 2 kernel time : 1460.227 ms
Throughput for kernel with max_concurrency 2: 0.561 GFlops
Max concurrency 4 kernel time : 1459.970 ms
Throughput for kernel with max_concurrency 4: 0.561 GFlops
Max concurrency 8 kernel time : 1460.034 ms
Throughput for kernel with max_concurrency 8: 0.561 GFlops
Max concurrency 16 kernel time : 1459.901 ms
Throughput for kernel with max_concurrency 16: 0.561 GFlops
PASSED: The results are correct
```

### Discussion of Results

The stdout output shows the giga-floating point operations per second (GFlops) for each kernel. 

When run on the Intel® PAC with Intel Arria10® 10 GX FPGA hardware board, we see that the throughput doubles from using max_concurrency 1 to max_concurrency 2, after which increasing the value of max_concurrency does not increase the GFlops achieved, i.e., increasing the max_concurrency above 2 will spend additional RAM resources for no additional throughput gain. As such, for this tutorial design, maximal throughput is best achieved by using max_concurrency 2. 

Using max_concurrency 0 (or equivalently omitting the attribute entirely) also produced good throughput, indicating that the compiler's default heuristic chose a concurrency of 2 or higher in this case.

When run on the FPGA emulator, the max_concurrency attribute has no effect on runtime. You may notice that the emulator achieved higher throughput than the FPGA in this example. This is because this trivial example uses only a tiny fraction of the spacial compute resources available on the FPGA.
