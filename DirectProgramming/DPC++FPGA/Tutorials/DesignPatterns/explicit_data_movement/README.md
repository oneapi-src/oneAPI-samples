# Explicit Data Movement
An FPGA tutorial demonstrating an alternative coding style, SYCL Unified Shared Memory (USM) device allocations, in which data movement between host and device is controlled explicitly by the code author.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | How to explicitly manage the movement of data for the FPGA
| Time to complete                  | 15 minutes



## Purpose
The purpose of this tutorial is to demonstrate an alternative coding style that allows you to explicitly control the movement of data in your SYCL program by using SYCL USM device allocations.

### Implicit vs. Explicit Data Movement
A typical SYCL design copies input data from the Host Memory to the FPGA's Device Memory. To do this, the data is sent to the FPGA board over PCIe. Once all the data is copied to the FPGA's Device Memory, the FPGA kernel is run and produces output that is also stored in Device Memory. Finally, the output data is transferred from the FPGA's Device Memory back to the CPU's Host Memory over PCIe. This model is shown in the figure below.

```
|-------------|                               |---------------|
|             |   |-------| PCIe |--------|   |               |
| Host Memory |<=>|  CPU  |<====>|  FPGA  |<=>| Device Memory |
|             |   |-------|      |--------|   |               |
|-------------|                               |---------------|
```

SYCL provides two methods for managing the momevement of data between host and device:
- *Implicit data movement*: Copying data to and from the FPGA's Device Memory is not controlled by the code author but rather by a parallel runtime. The parallel runtime is responsible for ensuring that data is correctly transferred before it is used and that kernels accessing the same data are synchronized appropriately.
- *Explicit data movement*: Copying data to and from the FPGA's Device Memory is explicitly handled by the code author in the source code. The code author is responsible for ensuring that data is correctly transferred before it is used and that kernels accessing the same data are synchronized appropriately.

Most of the other code sample tutorials in [DPC++FPGA/Tutorials](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA) use implicit data movement, which is achieved through the `buffer` and `accessor` SYCL constructs. This tutorial demonstrates an alternative coding style using explicit data movement, which is achieved through USM device allocations. 

### SYCL USM device allocations
SYCL USM device allocations enable the use of raw *C-style* pointers in SYCL, called *device pointers*. Allocating space in the FPGA's Device Memory for device pointers is done explicitly using the `malloc_device` function. Copying data to or from a device pointer must be done explicitly by the programmer, typically using the SYCL `memcpy` function. Additionally, synchronization between kernels accessing the same device pointers must be expressed explicitly by the programmer using either the `wait` function of a SYCL `event` or the `depends_on` signal between events.

In this tutorial, the `SubmitImplicitKernel` function demonstrates the basic SYCL buffer model, while the `SubmitExplicitKernel` function demonstrates how you may use these USM device allocations to perform the same task and manually manage the movement of data.

### Which Style Should I Use?
The main advantage of explicit data movement is that it gives you full control over when data is transferred between different memories. In some cases, we can improve our design's overall performance by overlapping data movement and computation, which can be done with more fine-grained control using USM device allocations. The main disadvantage of explicit data movement is that specifying all data movements can be tedious and error prone.

Choosing a data movement strategy largely depends on the specific application and personal preference. However, when starting a new design, it is typically easier to begin by using implicit data movement since all of the data movement is handled for you, allowing you to focus on expressing and validating the computation. Once the design is validated, you can profile your application. If you determine that there is still performance on the table, it may be worthwhile switching to explicit data movement.

Alternatively, there is a hybrid approach that uses some implicit data movement and some explicit data movement. This technique, demonstrated in the **Double Buffering** (double_buffering) and  **N-Way Buffering** (n_way_buffering) tutorials, uses implicit data movement for some buffers where the control does not affect performance, and explicit data movement for buffers whose movement has a substantial effect on performance. In this hybrid approach, we do **not** use device allocations but rather specific `buffer` API calls (e.g., `update_host`) to trigger the movement of data.
 
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
 
## Building the `explicit_data_movement` Tutorial
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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/explicit_data_movement.fpga.tar.gz" download>here</a>.
 
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

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide).

## Examining the Reports
Locate `report.html` in the `explicit_data_movement_report.prj/reports/` or `explicit_data_movement_report_s10_pac_report.prj/reports/report.html` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

## Running the Sample
 
 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./explicit_data_movement.fpga_emu          (Linux)
     explicit_data_movement.fpga_emu.exe        (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./explicit_data_movement.fpga              (Linux)
     ```

### Example of Output
You should see the following output in the console:

1. When running on the FPGA emulator
    ```
    Running the ImplicitKernel with size=10000
    Running the ExplicitKernel with size=10000
    PASSED
    ```

2. When running on the FPGA device
    ```
    Running the ImplicitKernel with size=100000000
    Running the ExplicitKernel with size=100000000
    Average latency for the ImplicitKernel: 530.762 ms
    Average latency for the ExplicitKernel: 470.453 ms
    PASSED
    ```
