# Zero-copy Data Transfer
This tutorial demonstrates how to use zero-copy host memory via the SYCL restricted Unified Shared Memory (USM) to improve the performance of your FPGA design.

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (Beta)
| What you will learn               | How to use Restricted USM for the FPGA
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_</br>
_Notice: Restricted USM (and therefore this tutorial) is only supported for the Intel® Programmable Acceleration Card (PAC) with Intel Stratix 10 SX FPGA_

## Purpose
The purpose of this tutorial is to show you how to take advantage of zero-copy host memory for the FPGA to improve the performance of your design. On an FPGA, all host and shared allocations are implemented as *zero-copy* data in host memory. This means that the FPGA will access the data directly over PCIe, which can improve performance in cases where there is little or no temporal reuse of data in the FPGA kernel. This tutorial includes two different kernels: one using traditional SYCL buffers (`src/buffer_kernel.hpp`) and one using restricted USM (`src/restricted_usm_kernel.hpp`) that takes advantage of zero-copy host memory. Before completing this tutorial, it is suggested you review the **Explicit USM** (explicit_usm) tutorial.

### Restricted USM
Restricted USM allows the host and device to share their respective memories. A typical SYCL design, which transfers data using either SYCL buffers/accessors or explicit USM, copies its input data from the Host Memory to the FPGA's Device Memory. To do this, the data is sent to the FPGA board over PCIe. Once all the data is copied to the FPGA's Device Memory, the FPGA kernel is run and produces output that is also stored in Device Memory. Finally, the output data is transferred from the FPGA's Device Memory back to the CPU's Host Memory over PCIe. This model is shown in the figure below.

```
|-------------|                                   |---------------|
|             |   |-------|   PCIe   |--------|   |               |
| Host Memory |<=>|  CPU  |<========>|  FPGA  |<=>| Device Memory |
|             |   |-------|          |--------|   |               |
|-------------|                                   |---------------|
```

Consider a kernel that simply performs computation for each entry in a buffer independently. Using SYCL buffers or explicit USM, we would bulk transfer the data from the Host Memory to the FPGA's Device Memory, run the kernel that performs the computation on each entry in the buffer, and then bulk transfer the buffer back to the host.

However, a better approach would simply stream the data from the host memory to the FPGA over PCIe, perform the computation on each piece of data, and then stream it back to host memory over PCIe. The desired structure is illustrated below. This would enable us to eliminate the overhead of copying the data to and from the Host Memory and the FPGA's Device Memory. This is done by using zero-copy host memory via the SYCL restricted USM model. This technique is demonstrated in `src/restricted_usm_kernel.hpp`.

```
|---------------|
|               |    |-------|  PCIe   |--------|
|               |    |       |========>|        |
|  Host Memory  |<==>|  CPU  |         |  FPGA  |
|               |    |       |<========|        |
|               |    |-------|  PCIe   |--------|
|---------------|
```

This approach is not considered host streaming, since the CPU and FPGA cannot (reliably) access the input/output data simultaneously. In other words, the host must wait until all the FPGA kernels have finished before (reliably) accessing the output data. However, we did avoid copying the data to and from the FPGA's Device Memory and therefore we get an overall savings in total latency. This savings can be seen by running the sample on FPGA hardware, or by the example output later in the [Example of Output](#example-of-output) section. In the future, we plan to create a tutorial that describes how to achieve true host streaming using restricted USM.

## Key Concepts
* How to use restricted USM for the FPGA
* The performance benefits of using restricted USM over traditional SYCL buffers or explicit USM
 
## License
This code sample is licensed under MIT license.
 
## Building the `zero_copy_data_transfer` Tutorial
### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.
 
### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile or fpga_runtime) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/get-started/base-toolkit/](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)).
 
When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.
 
### On a Linux* System
 
1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:  
    ```
    cmake ..
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
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Stratix® 10 SX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/zero_copy_data_transfer.fpga.tar.gz" download>here</a>.
 
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

   * Compiling for FPGA hardware is not yet supported on Windows.
 
### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide).

## Examining the Reports
Locate `report.html` in the `zero_copy_data_transfer_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

## Running the Sample
 
 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./zero_copy_data_transfer.fpga_emu     (Linux)
     zero_copy_data_transfer.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./zero_copy_data_transfer.fpga         (Linux)
     ```
 
### Example of Output
You should see the following output in the console:

1. When running on the FPGA emulator
    ```
    Running the buffer kernel version with size=10000
    Running the restricted USM kernel version with size=10000
    PASSED
    ```

2. When running on the FPGA device
    ```
    Running the buffer kernel with size=100000000
    Running the restricted USM kernel with size=100000000
    Average latency for the buffer kernel: 479.713 ms
    Average latency for the restricted USM kernel: 310.734 ms
    PASSED
    ```
