# `Board Test` Sample
The `Board Test` sample is a reference design contains tests to check FPGA board interfaces and reports the following metrics:

- Host to device global memory interface bandwidth
- Kernel clock frequency
- Kernel launch latency
- Kernel to device global memory bandwidth

| Area                    | Description
|:---                     |:---
| What you will learn     | How to test board interfaces to ensure that the designed platform provides expected performance
| Time to complete        | 30 minutes (not including compile time)

## Purpose
This reference design implements tests to check FPGA board interfaces and measure host-to-device and kernel-to-global memory interface metrics. Custom platform developers can use this reference design as a starting point to validate custom platform interfaces.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 4 sample that demonstrates a reference design.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")
   
   tier1 --> tier2 --> tier3 --> tier4
   
   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software                | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

> :warning: This sample is benchmarking an FPGA board, therefore it should really be used when targeting an FPGA board/BSP.

## Key Implementation Details

A oneAPI Board Support Package (BSP) consists of software layers and an FPGA hardware scaffold design, making it possible to target an FPGA through the Intel® oneAPI DPC++/C++ Compiler.

The compiler stitches the generated FPGA design into the oneAPI BSP framework. Refer to the [Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/programming-interface/fpga-flow/fpga-bsps-and-boards.html) for information about oneAPI BSPs.

The BSP hardware components typically comprise RTL for all interfaces the oneAPI kernel requires; for example, a PCIe IP for the host to kernel communication, EMIF (External Memory Interface) IP for kernel to memory, and host to FPGA board memory communication among other things.

The BSP software components typically consist of a Memory Mapped Device (MMD) layer and a driver. The implementation is vendor-dependent.

The BSP consists of components operating at different clock domains. PCIe and external memories operate at a fixed frequency. Corresponding RTL IPs are parametrized to operate at these fixed frequencies by platform vendors. The kernel clock frequency varies and is calculated as part of the oneAPI offline compilation flow for FPGAs. The BSP has logic to handle the data transfer across these clock domains.

`Board Test` measures the frequency that the kernel is running at in the FPGA and compares this to the compiled kernel clock frequency.

The following block diagram shows an overview of a typical oneAPI FPGA BSP hardware design and the numbered arrows depict the following:

- Path 1 represents the host-to-device global memory interface.
- Path 2 represents the host to kernel interface.
- Path 3 represents the kernel-to-device global memory interface.


![BSP hardware design](assets/oneapi_fpga_platform.png)

> **Note**: The block diagram shown is an overview of a typical oneAPI FPGA platform. See the oneAPI platform or BSP vendor documentation for more details about platform components.

### Source Code Explanation

| File               | Description
|:---                |:---
| `board_test.cpp`   | Contains the `main()` function and the test selection logic as well as calls to each test.
| `board_test.hpp`   | Contains the definitions for all the individual tests in the sample.
| `host_speed.hpp`   | Header for host speed test. Contains definition of functions used in host speed test.
| `helper.hpp`       | Contains constants (for example, binary name) used throughout the code as well as definition of functions that print help and measure execution time.

### Compiler Flags Used

| Flag                  | Description
|:---                   |:---
`-Xsno-interleaving`    | By default oneAPI compiler burst interleaves across same memory type.  `-Xsno-interleaving` disables burst interleaving and enables testing each memory bank independently. (See the [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/optimization-flags/disabl-burst-int.html) for more information.)

### Performance

Performance results are based on testing as of Jan 31, 2022.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

## Build the `Board Test` Program

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. 
> Set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation every time you open a new terminal window. 
> This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On Linux*

1. Change to the sample directory.
2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ``` 
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command: 
   >  ```
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ``` 
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `board_test.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Configure the build system for the Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ``` 
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command: 
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ``` 
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `board_test_report.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Board Test` Executable

The `Board Test` program checks following interfaces in a platform:

- **Host-to-device global memory interface:** This interface is checked by performing explicit data movement between the host and device global memory. Host to device global memory bandwidth is measured and reported. As a part of this interface check, unaligned data transfers are also performed to verify that non-DMA transfers complete successfully.

- **Kernel-to-device global memory interface:** This interface is checked by performing kernel to memory data transfers using simple read and write kernels. Kernel to memory bandwidth is measured and reported.

  > **Note**: This test currently does not support SYCL Unified Shared Memory (USM). For testing the USM interface, use the [Simple host streaming sample](/DirectProgramming/C++SYCL_FPGA/Tutorials/DesignPatterns/simple_host_streaming) code sample in the oneAPI-sample GitHub repository.

- **Host-to-kernel interface:** The test ensures that the host to kernel communication is correct and that the host can launch a kernel successfully. It also measures the roundtrip kernel launch latency and throughput (number of kernels/ms) of single task no-operation kernels.

- **Kernel clock frequency:** The test measures the frequency the programmed kernel is running at on the FPGA device and reports it. By default, this test fails if the measured frequency is not within 2% of the compiled frequency.

  >**Note**: Kernel clock frequency test measures the frequency that the programmed kernel is running at on the FPGA device and reports it. By default, this test fails if the measured frequency is not within 2% of the compiled frequency. The test allows overriding this failure; however, overriding might lead to functional errors, and it is not recommended. The override option is provided to allow debug in case where platform design changes are done to force kernel to run at slower clock (this is not a common use-case). To override, set the `report_chk` variable to `false` in `board_test.cpp` and recompile only the host code by using the `-reuse-exe=board_test.fpga` option in your compile command.


### Configurable Parameters

The complete board test is divided into six subtests. By default, all tests run. You can choose to run a single test by using the `-test=<test number>` option. Refer to the [Running the Sample](#running-the-sample) section for test usage instructions.

| Test Number  | Test Name
|:---          |:---
| 1            | Host Speed and Host Read Write Test
| 2            | Kernel Clock Frequency Test
| 3            | Kernel Launch Test
| 4            | Kernel Latency Measurement
| 5            | Kernel-to-Memory Read Write Test
| 6            | Kernel-to-Memory Bandwidth Test

>**Note:** You should run all tests at least once to ensure that the platform interfaces are fully functional.

To view test details and usage information using the binary, use the `-help` option: `<program> -help`.

### On Linux

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    ./board_test.fpga_emu
    ```
 2. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
    ```
    ./board_test.fpga
    ```
    By default the program runs all tests. To run a specific test, enter the test number as an argument to the `-test` option:
    ```
    ./board_test.fpga -test=<test_number>
    ```
### On Windows

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
    ```
    board_test.exe            (Windows)
    ```
    By default the program runs all tests. To run a specific test, enter the test number as an argument to the `-test` option:
    ```
    board_test.exe -test=<test_number>
    ```
 2. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
    ```
    ./board_test.fpga.exe
    ```
    By default the program runs all tests. To run a specific test, enter the test number as an argument to the `-test` option:
    ```
    ./board_test.fpga.exe -test=<test_number>
    ```

## Example Output

Running on FPGA device (Intel Stratix 10 SX platform).

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

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
