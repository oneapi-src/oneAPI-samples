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

Running on FPGA device (Intel® FPGA SmartNIC N6001-PL). Performance results are based on testing as of May 10, 2024.

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
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)

clGetDeviceInfo CL_DEVICE_GLOBAL_MEM_SIZE = 17179869184
clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE = 17179868160
Device buffer size available for allocation = 17179868160 bytes

*****************************************************************
*********************** Host Speed Test *************************
*****************************************************************

Size of buffer created = 17179868160 bytes
Writing 16383 MiB to device global memory ... 7592.7 MB/s
Reading 16383 MiB from device global memory ... 7628.8 MB/s
Verifying data ...
Successfully wrote and readback 16383 MB buffer

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
   32768 381.23 426.21 249.94 4775.26 
   65536 510.22 546.47 406.00 7332.94 
  131072 757.51 1073.87 701.72 13826.89 
  262144 977.82 1954.74 869.33 16369.93 
  524288 1272.50 3282.34 1037.37 14452.23 
 1048576 1746.77 4678.85 1083.52 8202.39 
 2097152 5797.93 5983.74 5546.83 20416.05 
 4194304 6436.09 6557.66 6318.95 12325.60 
 8388608 6919.11 6919.11 6919.11 6919.11 

Reading 8192 KBs with block size (in bytes) below:

Block_Size Avg Max Min End-End (MB/s)
   32768 416.23 463.80 149.57 4051.34 
   65536 588.92 634.71 261.48 6861.12 
  131072 769.07 1089.64 397.59 12046.73 
  262144 1017.03 2195.03 654.95 16790.08 
  524288 1204.36 3581.23 815.31 13943.92 
 1048576 1512.05 4862.33 953.71 8371.75 
 2097152 2775.19 6196.34 1046.06 4133.37 
 4194304 2893.07 6699.52 1844.87 3673.20 
 8388608 2977.26 2977.26 2977.26 2977.26 

Host write top speed = 20416.05 MB/s
Host read top speed = 16790.08 MB/s


HOST-TO-MEMORY BANDWIDTH = 18603 MB/s


*****************************************************************
********************* Host Read Write Test **********************
*****************************************************************

--- Running host read write test with device offset 0
--- Running host read write test with device offset 3

HOST READ-WRITE TEST PASSED!

*****************************************************************
*******************  Kernel Clock Frequency Test  ***************
*****************************************************************

Measured Frequency    =   511.062 MHz 
Quartus Compiled Frequency  =   512 MHz 

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

Processed 10000 kernels in 118.6319 ms
Single kernel round trip time = 11.8632 us
Throughput = 84.2943 kernels/ms
Kernel execution is complete

*****************************************************************
*************  Kernel-to-Memory Read Write Test  ***************
*****************************************************************

Maximum device global memory allocation size is 17179868160 bytes 
Finished host memory allocation for input and output data
Creating device buffer
Finished writing to device buffers 
Launching kernel MemReadWriteStream ... 
Launching kernel with global offset : 0
Launching kernel with global offset : 1073741824
Launching kernel with global offset : 2147483648
Launching kernel with global offset : 3221225472
... kernel finished execution. 
Finished Verification
KERNEL TO MEMORY READ WRITE TEST PASSED 

*****************************************************************
*****************  Kernel-to-Memory Bandwidth  *****************
*****************************************************************

Note: This test assumes that design was compiled with -Xsno-interleaving option


Performing kernel transfers of 4096 MiBs on the default global memory (address starting at 0)
Launching kernel MemWriteStream ... 
Launching kernel MemReadStream ... 
Launching kernel MemReadWriteStream ... 

Summarizing bandwidth in MB/s/bank for banks 1 to 8
 8765.24  8765.28  8765.26  8765.29  8765.27  8765.24  8765.3  8765.28  MemWriteStream
 8786.28  8786.28  8786.27  8786.26  8786.27  8786.26  8786.3  8786.26  MemReadStream
 8059.25  8062.61  8061.29  8054.25  8058.78  8061.35  8062.6  8058.39  MemReadWriteStream

KERNEL-TO-MEMORY BANDWIDTH = 8537.12 MB/s/bank

*****************************************************************
***********************  USM Bandwidth  *************************
*****************************************************************

Board does not support USM, skipping this test.

BOARD TEST PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
