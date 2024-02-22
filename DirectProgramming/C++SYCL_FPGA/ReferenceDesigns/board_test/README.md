# `Board Test` Sample
The `Board Test` sample is a reference design that contains tests to check FPGA board interfaces and reports the following metrics:

- Host-to-device global memory interface bandwidth
- Kernel clock frequency
- Kernel launch latency
- Kernel-to-device global memory bandwidth
- Unified Shared Memory bandwidth

| Area                    | Description
|:---                     |:---
| What you will learn     | How to test board interfaces to ensure that the designed platform provides expected performance
| Time to complete        | 30 minutes (not including compile time)

## Purpose
This reference design implements tests to check FPGA board interfaces and measure host-to-device and kernel-to-global memory interface metrics. Use this reference design as a starting point to validate platform interfaces when you customize a BSP.

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
| OS                      | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware                | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software                | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
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

- Path 1 represents the host to kernel interface.
- Path 2 represents the host-to-device global memory interface.
- Path 3 represents the kernel-to-device global memory interface.
- Path 4 represents the kernel-to-shared host memory interface


![BSP hardware design](assets/oneapi_fpga_platform.png)

> **Note**: The block diagram shown is an overview of a typical oneAPI FPGA platform. See the oneAPI platform or BSP vendor documentation for more details about platform components.

### Source Code Explanation

| File               | Description
|:---                |:---
| `board_test.cpp`   | Contains the `main()` function and the test selection logic as well as calls to each test.
| `board_test.hpp`   | Contains the definitions for all the individual tests in the sample.
| `host_speed.hpp`   | Header for host speed test. Contains definition of functions used in host speed test.
| `usm_speed.hpp`    | Header for the USM bandwidth test. Contains definitions of functions used in the USM bandwidth test.
| `helper.hpp`       | Contains constants (for example, binary name) used throughout the code as well as definition of functions that print help and measure execution time.

### Compiler Flags Used

| Flag                  | Description
|:---                   |:---
`-Xsno-interleaving`    | By default oneAPI compiler burst interleaves across same memory type.  `-Xsno-interleaving` disables burst interleaving and enables testing each memory bank independently. (See the [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/optimization-flags/disabl-burst-int.html) for more information.)

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
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On Linux*

1. Change to the sample directory.
2. Configure the build system for your BSP.

   ```
   mkdir build
   cd build
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
   > **Note**: You must set FPGA_DEVICE to point to your BSP in order to build this sample.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the optimization report. 
      ```
      make report
      ```
   3. Compile and run for FPGA hardware (longer compile time, targets an FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Configure the build system for your BSP.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
   > **Note**: You must set FPGA_DEVICE to point to your BSP in order to build this sample.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report.
      ```
      nmake report
      ```
   3. Compile and run for FPGA hardware (longer compile time, targets an FPGA device).
      ```
      nmake fpga
      ```
>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your 'build' directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory, for example:
>
>  ```
  > C:\samples\build> cmake -G "NMake Makefiles" C:\long\path\to\code\sample\CMakeLists.txt
>  ```
## Run the `Board Test` Executable

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
| 7            | Unified Shared Memory (USM) Bandwidth Test

>**Note:** You should run all tests at least once to ensure that the platform interfaces are fully functional.

To view test details and usage information using the binary, use the `-help` option: `<program> -help`.

The tests listed above check the following interfaces in a platform:

- **Host-to-device global memory interface (Test 1):** This interface is checked by performing explicit data movement between the host and device global memory. Host to device global memory bandwidth is measured and reported. As a part of this interface check, unaligned data transfers are also performed to verify that non-DMA transfers complete successfully.

- **Kernel clock frequency (Test 2):** The test measures the frequency the programmed kernel is running at on the FPGA device and reports it. By default, this test fails if the measured frequency is not within 2% of the compiled frequency.

  >**Note**: The test allows overriding this failure; however, overriding may lead to functional errors and is not recommended. The override option is provided to allow debugging in cases where platform design changes are done to force the kernel to run at a slower clock frequency (though this is not a common use case). To override, set the `report_chk` variable to `false` in `board_test.cpp` and recompile only the host code by using the `-reuse-exe=board_test.fpga` option in your compile command (this flag is added by default for you in the CMake file included with this code sample).

- **Host-to-kernel interface (Tests 3 & 4):** The test ensures that the host to kernel communication is correct and that the host can launch a kernel successfully. It also measures the roundtrip kernel launch latency and throughput (number of kernels/ms) of single task no-operation kernels.

- **Kernel-to-device global memory interface (Tests 5 & 6):** This interface is checked by performing kernel to memory data transfers using simple read and write kernels. Kernel to memory bandwidth is measured and reported.

- **Unified shared memory (USM) interface (Test 7):** This interface is checked by copying data between, reading data from, and writing data to host USM. The bandwidth is measured and reported for each case. Applies only to board variants with USM support; to run this test you must specify the `SUPPORTS_USM` macro at compile-time; e.g., `cmake .. -DSUPPORTS_USM=1`.

### On Linux

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    ./board_test.fpga_emu
    ```
    By default the program runs all tests. To run a specific test, enter the test number as an argument to the `-test` option:
    ```
    ./board_test.fpga_emu -test=<test_number>
    ```
 2. Run the sample on the FPGA device.
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
    board_test.exe
    ```
    By default the program runs all tests. To run a specific test, enter the test number as an argument to the `-test` option:
    ```
    board_test.exe -test=<test_number>
    ```
 2. Run the sample on the FPGA device.
    ```
    board_test.fpga.exe
    ```
    By default the program runs all tests. To run a specific test, enter the test number as an argument to the `-test` option:
    ```
    board_test.fpga.exe -test=<test_number>
    ```

## Example Output

Running on FPGA device (Terasic’s DE10-Agilex Development Board). Performance results are based on testing as of August 30, 2023.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

```
*** Board_test usage information ***
Command to run board_test using generated binary:
  > To run all tests (default): run board_test.fpga
  > To run a specific test (see list below); pass the test number as argument to "-test" option: 
  Linux: ./board_test.fpga -test=<test_number>
  Windows: board_test.exe -test=<test_number>
  > To see more details on what each test does use -help option
The tests are:
  1. Host Speed and Host Read Write Test
  2. Kernel Clock Frequency Test
  3. Kernel Launch Test
  4. Kernel Latency Measurement
  5. Kernel-to-Memory Read Write Test
  6. Kernel-to-Memory Bandwidth Test
  7. Unified Shared Memory Bandwidth Test
Note: Kernel Clock Frequency is run along with all tests except 1 (Host Speed and Host Read Write test)

Running all tests 
Running on device: de10_agilex : Agilex Reference Platform (aclde10_agilex0)

clGetDeviceInfo CL_DEVICE_GLOBAL_MEM_SIZE = 34359737344
clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE = 34359737344
Device buffer size available for allocation = 34359737344 bytes

*****************************************************************
*********************** Host Speed Test *************************
*****************************************************************

Size of buffer created = 34359737344 bytes
Writing 32767 MB to device global memory ... 8776.42 MB/s
Reading 32767 MB from device global memory ... 9743.93 MB/s
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
   32768 428.07 435.44 382.82 395.97 
   65536 783.64 793.81 665.47 730.92 
  131072 1325.34 1343.33 1165.79 1250.47 
  262144 1984.22 2016.88 1776.27 1903.43 
  524288 3507.84 3588.65 3165.04 3385.12 
 1048576 4845.12 4982.59 4533.71 4730.02 
 2097152 5741.51 5758.96 5719.79 5656.95 
 4194304 6695.96 6869.04 6531.39 6652.06 
 8388608 7585.54 7585.54 7585.54 7585.54 

Reading 8192 KBs with block size (in bytes) below:

Block_Size Avg Max Min End-End (MB/s)
   32768 477.89 492.37 430.08 436.84 
   65536 869.03 896.61 814.20 799.03 
  131072 1464.26 1504.16 1384.23 1363.57 
  262144 2201.40 2237.72 2136.42 2090.72 
  524288 3869.46 3966.81 3728.03 3697.54 
 1048576 5318.21 5457.59 5197.05 5171.01 
 2097152 6325.13 6432.15 6175.55 6217.27 
 4194304 7577.67 7609.52 7546.07 7526.88 
 8388608 8441.18 8441.18 8441.18 8441.18 

Host write top speed = 7585.54 MB/s
Host read top speed = 8441.18 MB/s


HOST-TO-MEMORY BANDWIDTH = 8013 MB/s


*****************************************************************
********************* Host Read Write Test **********************
*****************************************************************

--- Running host read write test with device offset 0
** WARNING: [aclde10_agilex0] NOT using DMA to transfer 1024 bytes from host to device because of lack of alignment
**                 host ptr (0x1688a3f5) and/or dev offset (0x400) is not aligned to 4 bytes
--- Running host read write test with device offset 3
** WARNING: [aclde10_agilex0] NOT using DMA to transfer 1024 bytes from host to device because of lack of alignment
**                 host ptr (0x1688a3f5) and/or dev offset (0x403) is not aligned to 4 bytes
** WARNING: [aclde10_agilex0] NOT using DMA to transfer 1024 bytes from device to host because of lack of alignment
**                 host ptr (0x16893cb8) and/or dev offset (0x403) is not aligned to 4 bytes

HOST READ-WRITE TEST PASSED!

*****************************************************************
*******************  Kernel Clock Frequency Test  ***************
*****************************************************************

Measured Frequency    =   598.905 MHz 
Quartus Compiled Frequency  =   600 MHz 

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

Processed 10000 kernels in 217.0312 ms
Single kernel round trip time = 21.7031 us
Throughput = 46.0763 kernels/ms
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
 19307.6  19312.5  19309.3  19309.4  19309.2  19309.4  19311.3  19309.2  MemWriteStream
 19337.7  19339.8  19337.7  19341.4  19340.3  19338.3  19337.7  19339.3  MemReadStream
 17657.3  17657.1  17657.5  17657.4  17656.7  17657.7  17657.6  17657.5  MemReadWriteStream

KERNEL-TO-MEMORY BANDWIDTH = 18768.7 MB/s/bank

*****************************************************************
***********************  USM Bandwidth  *************************
*****************************************************************

Board does not support USM, skipping this test.

BOARD TEST PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
