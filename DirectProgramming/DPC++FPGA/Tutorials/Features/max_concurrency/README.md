# Maximum Concurrency of a Loop
This FPGA tutorial explains how to use the max_concurrency attribute for loops.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | The basic usage of the `max_concurrency` attribute <br> How the `max_concurrency` attribute affects loop throughput and resource use <br> How to apply the `max_concurrency` attribute to loops in your program <br> How to identify the correct `max_concurrency` factor for your program
| Time to complete                  | 15 minutes



## Purpose
This tutorial demonstrates a simple example of applying the `max_concurrency` attribute to a loop in a task kernel to trade off the on-chip memory use and throughput of the loop.

### Description of the `max_concurrency` Attribute
The `max_concurrency` attribute is a loop attribute that enables you to control the number of simultaneously executed loop iterations. To enable simultaneous execution, the compiler creates copies of any memory with single iteration scope. These copies are called private copies. The greater the permitted concurrency, the more private copies the compiler must create. 

#### Example: 

Kernels in this tutorial design apply `[[intelfpga::max_concurrency(N)]]` to an outer loop that contains two inner loops, which perform a partial sum computation on an input array, storing the results in a private (to the outer loop) array `a1`. The following is an example of a loop nest:

```
[[intelfpga::max_concurrency(1)]]
for (size_t i = 0; i < max_iter; i++) {                                                      
  float a1[size];                                                                              
  for (int j = 0; j < size; j++)                                                               
    a1[j] = accessor_a[i * 4 + j] * shift;                                                      
  for (int j = 0; j < size; j++)                                                               
    result += a1[j];                                                                           
}   
```

In this example, the maximum concurrency allowed for the outer loop is 1. Only one iteration of the outer loop is allowed to be simultaneously executing at any given moment. The `max_concurrency` attribute in this example forces the compiler to create one private copy of the array `a1`. Passing the parameter `N` to the `max_concurrency` attribute limits the loop's concurrency to `N` simultaneous iterations and `N` private copies of privately-declared arrays in that loop.

### Identifying the Correct `max_concurrency` Factor
Generally, increasing the maximum concurrency allowed for a loop through the use of the `max_concurrency` attribute increases the throughput of that loop at the cost of increased memory resource use. Additionally, in nearly all cases, there is a point at which increasing the maximum concurrency does not have any further effect on the throughput of the loop, as the maximum exploitable concurrency of that loop has been achieved. 

The correct `max_concurrency` factor for a loop depends on your goals for the design, the criticality of the loop in question, and its impact on your design's overall throughput. A typical design flow may be to: 
1. Experiment with different values of `max_concurrency`. 
2. Observe what impact the values have on the overall throughput and memory use of your design.
3. Choose the appropriate value that allows you to achieve your desired throughput and area goals.

## Key Concepts
* The basic usage of the `max_concurrency` attribute 
* How the `max_concurrency` attribute affects loop throughput and resource use
* How to apply the `max_concurrency` attribute to loops in your program
* How to identify the correct `max_concurrency` factor for your program

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `max_concurrency` Tutorial

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
3. (Optional) As the FPGA hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/max_concurrency.fpga.tar.gz" download>here</a>.

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
Max concurrency 0 kernel time : 1457.47 ms
Throughput for kernel with max_concurrency 0: 562 MIPS
Max concurrency 1 kernel time : 2947.784 ms
Throughput for kernel with max_concurrency 1: 278 MIPS
Max concurrency 2 kernel time : 1471.743 ms
Throughput for kernel with max_concurrency 2: 557 MIPS
Max concurrency 4 kernel time : 1457.460 ms
Throughput for kernel with max_concurrency 4: 562 MIPS
Max concurrency 8 kernel time : 1457.461 ms
Throughput for kernel with max_concurrency 8: 562 MIPS
Max concurrency 16 kernel time : 1457.463 ms
Throughput for kernel with max_concurrency 16: 562 MIPS
PASSED: The results are correct
```

### Discussion of Results

The stdout output shows the million instructions per second (MIPS) for each kernel. 

When run on the Intel® PAC with Intel Arria10® 10 GX FPGA hardware board, we see that the throughput doubles from using max_concurrency 1 to max_concurrency 2. Further increasing the value of max_concurrency does not increase the MIPS achieved, i.e., increasing the max_concurrency above 2 will spend additional RAM resources for no additional throughput gain. As such, for this tutorial design, maximal throughput is best achieved by using max_concurrency 2. 

Using max_concurrency 0 (or equivalently omitting the attribute entirely) also produced good throughput, indicating that the compiler's default heuristic chose a concurrency of 2 or higher in this case.

When run on the FPGA emulator, the max_concurrency attribute has no effect on runtime. You may notice that the emulator achieved higher throughput than the FPGA in this example. This is because this trivial example uses only a tiny fraction of the spatial compute resources available on the FPGA.
