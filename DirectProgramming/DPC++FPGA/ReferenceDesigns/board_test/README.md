# Board Test
This reference design contains tests to check FPGA board interfaces and reports the following metrics:

* Host to device global memory interface bandwidth
* Kernel clock frequency
* Kernel launch latency
* Kernel to device global memory bandwidth

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15, Windows 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | How to test board interfaces to ensure the designed platform provides expected performance
| Time to complete                  | 30 minutes (not including compile time)

## Purpose
This reference design implements tests to check FPGA board interfaces and measure host to device and kernel to global memory interface metrics. Custom platform developers can use this test as a starting point to validate custom platform interfaces.

### Introduction to oneAPI FPGA Board Support Package(BSP)

A oneAPI board support package (BSP) consists of software layers and an FPGA hardware scaffold design, making it possible to target an FPGA through the Intel® oneAPI DPC++/C++ Compiler. The compiler stitches the generated FPGA design into the oneAPI BSP framework. Refer to the [oneAPI Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/programming-interface/fpga-flow/fpga-bsps-and-boards.html) for information about oneAPI BSPs.

The BSP hardware components typically comprise RTL for all interfaces the oneAPI kernel requires, for example, a PCIe IP for the host to kernel communication, EMIF (External Memory Interface) IP for kernel to memory and host to FPGA board memory communication, and so on.

The BSP software components typically consist of a Memory Mapped Device (MMD) layer and a driver. The implementation is vendor-dependent.

The following block diagram shows an overview of a typical oneAPI FPGA BSP hardware design and the numbered arrows depict the following:
* Path 1 represents the host-to-device global memory interface.
* Path 2 represents the host to kernel interface.
* Path 3 represents the kernel-to-device global memory interface.

The BSP consists of components operating at different clock domains. PCIe and external memories operate at a fixed frequency. Corresponding RTL IPs are parametrized to operate at these fixed frequencies by platform vendors.
The kernel clock frequency varies and is calculated as part of the oneAPI offline compilation flow for FPGAs. The BSP has logic to handle the data transfer across these clock domains.

`board_test` measures the frequency the kernel is running at in the FPGA and compares this to the compiled kernel clock frequency. [Test Details](#test-details) section provides more information.

***Note:*** The block diagram shown below is an overview of a typical oneAPI FPGA platform. Refer to the oneAPI platform or BSP vendor documentation for more details about platform components.

![](oneapi_fpga_platform.PNG)

### Test Details
The complete test exercises following interfaces in a platform:
* **Host to device global memory interface:** This interface is checked by performing explicit data movement between the host and device global memory. Host to device global memory bandwidth is measured and reported. As a part of this interface check, unaligned data transfers are also performed to verify that non-DMA transfers complete successfully.

* **Kernel to device global memory interface:** This interface is checked by performing kernel to memory data transfers using simple read and write kernels. Kernel to memory bandwidth is measured and reported.<br> 
  ***Note:*** This test currently does not support SYCL Unified Shared Memory (USM). For testing the USM interface, use the [Simple host streaming sample](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/Tutorials/DesignPatterns/simple_host_streaming) code sample.

* **Host to kernel interface:** The test ensures the host to kernel communication is correct and that the host can launch a kernel successfully. It also measures the roundtrip kernel launch latency and throughput (number of kernels/ms) of single task no-operation kernels.

In addition to the above interface checks, the test also measures **Kernel clock frequency**. The test measures the frequency the programmed kernel is running at on the FPGA device and reports it. By default, this test fails if the measured frequency is not within 2% of the compiled frequency.<br>
***Note:*** The test allows overriding this failure. However, this may lead to functional errors, and Intel does not recommend it. The override option is provided to allow debug in case where platform design changes are done to force kernel to run at slower clock (this is not a common use-case). The [Running the Sample section](#running-the-sample) describes how to override this option if required.

The complete board test is divided into six subtests. By default, all tests run. You can choose to run a single test by using the `-test=<test number>` option. Refer to the [Running the Sample](#running-the-sample) section for test usage instructions.
1. Host Speed and Host Read Write Test
2. Kernel Clock Frequency Test
3. Kernel Launch Test
4. Kernel Latency Measurement
5. Kernel-to-Memory Read Write Test
6. Kernel-to-Memory Bandwidth Test<br>
   
***Note:*** Intel recommends running all tests at least once to ensure that the platform interfaces are fully functional.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `board_test` code sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).
### Include Files
The included header `dpc_common.hpp` is located at `$ONEAPI_ROOT/dev-utilities/latest/include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 24h.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
   ```
   mkdir build
   cd build
   ```
   To compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:
   ```
    cmake ..
   ```
   Alternatively, to compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:
   ```
   cmake -DFPGA_BOARD=intel_a10gx_pac:pac_a10 ..
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -DFPGA_BOARD=<board-support-package>:<board-variant> ..
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (compiles quickly, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Generate the optimization report:
     ```
     make report
     ```
   * Compile for FPGA hardware (takes longer to compile, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, FPGA pre-compiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/board_test.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
   ```
   mkdir build
   cd build
   ```
   To compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:
   ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" -DFPGA_BOARD=intel_a10gx_pac:pac_a10 ..
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" -DFPGA_BOARD=<board-support-package>:<board-variant> ..
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (compiles quickly, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report:
     ```
     nmake report
     ```
   * An FPGA hardware target is not provided on Windows*.

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

If an error occurs, you can get more details by running make with the VERBOSE=1 argument: make VERBOSE=1 For more comprehensive troubleshooting, use the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors. [Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this Reference Design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html)

## Examining the Reports
Locate `report.html` in the `board_test.prj/reports/` directory and open the report in one of the following browsers: Chrome*, Firefox*, Edge*, or Internet Explorer*

You can also examine the reports generated by the full FPGA hardware compile and compare their contents.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./board_test.fpga_emu     (Linux)
     board_test.exe            (Windows)
     ```
 2. Run the sample on the FPGA device:
     ```
     ./board_test.fpga         (Linux)
     ```
 The above commands run all tests described in [Test Details section](#test-details). To run a specific test, pass the test number as an argument to the `-test=<test number>` option with the executable:
 ```
 ./board_test.fpga -test=<test_number>     (Linux)
 board_test.exe -test=<test_number>        (Windows)
 ```

The `test_number` should be one of the following:
1. Host Speed and Host Read Write Test
2. Kernel Clock Frequency Test
3. Kernel Launch Test
4. Kernel Latency Measurement
5. Kernel-to-Memory Read Write Test
6. Kernel-to-Memory Bandwidth Test

*Note:* Kernel clock frequency test measures the frequency that the programmed kernel is running at on the FPGA device and reports it. By default, this test fails if the measured frequency is not within 2% of the compiled frequency. The test allows overriding this failure. However, this may lead to functional errors, and Intel does not recommend it. The override option is provided to allow debug in case where platform design changes are done to force kernel to run at slower clock (this is not a common use-case). To override, set the `report_chk` variable to `false` in `board_test.cpp` and recompile only the host code by using the `-reuse-exe=board_test.fpga` option in your compile command.

To view test details and usage information using the binary, use the `-help` option.
```
./board_test.fpga -help      (Linux)
board_test.exe -help         (Windows)
```

### Example of Output
Running on FPGA device (Intel Stratix 10 SX platform):
```
*** Board_test usage information ***
Command to run board_test using generated binary:
  > To run all tests (default): ./board_test.fpga
  > To run a specific test (see list below); pass the test number as argument to "-test" option: ./board_test.fpga -test=<test_number>
  > To see more details on what each test does: ./board_test.fpga -help
The tests are:
  1. Host Speed and Host Read Write Test
  2. Kernel Clock Frequency Test
  3. Kernel Launch Test
  4. Kernel Latency Measurement
  5. Kernel-to-Memory Read Write Test
  6. Kernel-to-Memory Bandwidth Test
Note: Kernel Clock Frequency is run along with all tests except 1 (Host Speed and Host Read Write test)

Running all tests 
Running on device: pac_s10 : Intel PAC Platform (pac_ee00000)

clGetDeviceInfo CL_DEVICE_GLOBAL_MEM_SIZE = 34359737344
clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE = 34359737344
Device buffer size available for allocation = 34359737344 bytes

*****************************************************************
*********************** Host Speed Test *************************
*****************************************************************

Size of buffer created = 34359737344 bytes
Writing 32767 MB to device global memory ... 7462.91 MB/s
Reading 32767 MB from device global memory ... 6188.28 MB/s
Verifying data ...
Successfully wrote and readback 32767 MB buffer

Transferring 8192 KBs in 256 32 KB blocks ...
Transferring 8192 KBs in 128 64 KB blocks ...
Transferring 8192 KBs in 64 128 KB blocks ...
Transferring 8192 KBs in 32 256 KB blocks ...
Transferring 8192 KBs in 16 512 KB blocks ...
Transferring 8192 KBs in 8 1024 KB blocks ...
Transferring 8192 KBs in 4 2048 KB blocks ...
Transferring 8192 KBs in 2 4096 KB blocks ...
Transferring 8192 KBs in 1 8192 KB blocks ...

Writing 8192 KBs with block size (in bytes) below:

Block_Size Avg Max Min End-End (MB/s)
   32768 429.32 1087.94 26.37 407.51 
   65536 659.07 1602.89 61.65 634.23 
  131072 821.27 2180.32 215.22 801.83 
  262144 1645.40 2791.99 422.74 1544.55 
  524288 2183.01 3152.49 1173.56 2155.32 
 1048576 3129.02 3626.10 2356.49 3102.71 
 2097152 3433.38 3854.89 3163.94 3417.06 
 4194304 3973.04 4163.90 3798.91 3964.90 
 8388608 4412.13 4412.13 4412.13 4412.13 

Reading 8192 KBs with block size (in bytes) below:

Block_Size Avg Max Min End-End (MB/s)
   32768 837.41 1369.71 479.84 756.23 
   65536 1280.98 1979.54 714.56 1185.66 
  131072 1951.20 2610.64 1372.24 1841.02 
  262144 2398.59 3006.94 629.56 2309.85 
  524288 3273.74 3690.34 3059.41 3197.51 
 1048576 3586.16 3809.06 3421.94 3544.91 
 2097152 3630.12 3795.49 3441.44 3610.78 
 4194304 4330.42 4443.38 4223.06 4324.10 
 8388608 4659.93 4659.93 4659.93 4659.93 

Host write top speed = 4412.13 MB/s
Host read top speed = 4659.93 MB/s


HOST-TO-MEMORY BANDWIDTH = 4536 MB/s


*****************************************************************
********************* Host Read Write Test **********************
*****************************************************************

--- Running host read write test with device offset 0
--- Running host read write test with device offset 3

HOST READ-WRITE TEST PASSED!

*****************************************************************
*******************  Kernel Clock Frequency Test  ***************
*****************************************************************

Measured Frequency    =   330.542 MHz 
Quartus Compiled Frequency  =   331 MHz 

Measured Clock frequency is within 2 percent of Quartus compiled frequency. 

*****************************************************************
********************* Kernel Launch Test ************************
*****************************************************************

Launching kernel KernelSender ...
Launching kernel KernelReceiver ...
  ... Waiting for sender
Sender sent the token to receiver
  ... Waiting for receiver

KERNEL_LAUNCH_TEST PASSED

*****************************************************************
********************  Kernel Latency  **************************
*****************************************************************

Processed 10000 kernels in 149.3585 ms
Single kernel round trip time = 14.9359 us
Throughput = 66.9530 kernels/ms
Kernel execution is complete

*****************************************************************
*************  Kernel-to-Memory Read Write Test  ***************
*****************************************************************

Maximum device global memory allocation size is 34359737344 bytes 
Finished host memory allocation for input and output data
Creating device buffer
Finished writing to device buffers 
Launching kernel MemReadWriteStream ... 
Launching kernel with global offset : 0
Launching kernel with global offset : 1073741824
Launching kernel with global offset : 2147483648
Launching kernel with global offset : 3221225472
Launching kernel with global offset : 4294967296
Launching kernel with global offset : 5368709120
Launching kernel with global offset : 6442450944
Launching kernel with global offset : 7516192768
... kernel finished execution. 
Finished Verification
KERNEL TO MEMORY READ WRITE TEST PASSED 

*****************************************************************
*****************  Kernel-to-Memory Bandwidth  *****************
*****************************************************************

Note: This test assumes that design was compiled with -Xsno-interleaving option


Performing kernel transfers of 4096 MBs on the default global memory (address starting at 0)
Launching kernel MemWriteStream ... 
Launching kernel MemReadStream ... 
Launching kernel MemReadWriteStream ... 

Summarizing bandwidth in MB/s/bank for banks 1 to 8
 16138.4  16137.4  16137.6  16138.5  16138.4  16137.3  16137.4  16138.1  MemWriteStream
 17341.3  17341.2  17341.2  17341.1  17341.1  17341.1  17341.2  17341.1  MemReadStream
 16050.6  16049.9  16049.5  16049.5  16049.3  16049.7  16049.8  16049.5  MemReadWriteStream

KERNEL-TO-MEMORY BANDWIDTH = 16509.6 MB/s/bank

BOARD TEST PASSED
 
```
## Additional Design Information

### Source Code Explanation

| File                         | Description
---                            |---
| `board_test.cpp`             | Contains the `main()` function and the test selection logic as well as calls to each test.
| `board_test.hpp`             | Contains the definitions for all the individual tests in board_test
| `host_speed.hpp`             | Header for host speed test; contains definition of functions used in host speed test
| `helper.hpp`                 | Contains constants (e.g. binary name) used throughout the code as well as definition of functions that print help and measure execution time

### Compiler Flags Used

| Flag | Description
---    |---
`-Xsno-interleaving` | By default oneAPI compiler burst interleaves across same memory type, `-Xsno-interleaving` is used to disable burst interleaving and test each memory bank independently. Refer to [oneAPI DPC++ FPGA Optimization Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/optimization-flags/disabl-burst-int.html) for more information
### Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).

Performance results are based on testing as of Jan 31, 2022 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](https://www.intel.com).

The performance was measured by Intel on Jan 31, 2022.

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.