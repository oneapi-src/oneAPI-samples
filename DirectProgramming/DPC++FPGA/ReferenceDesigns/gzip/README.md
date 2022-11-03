# `GZIP` Sample

This reference design demonstrates high-performance GZIP compression on FPGA.

| Optimized for        | Description
|:---                  |:---
| What you will learn  | How to implement a high-performance multi-engine compression algorithm on FPGA
| Time to complete     | 1 hr (not including compile time)
| Category             | Reference Designs and End to End


## Purpose

This reference design implements a compression algorithm. The implementation is optimized for the FPGA device. The compression result is GZIP-compatible and can be decompressed with GUNZIP. The GZIP output file format is compatible with the GZIP DEFLATE algorithm and follows a fixed subset (see RFC 1951). (See the [Additional References](#additional-references) section for specific references.)

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler <br> Intel® FPGA Add-on for oneAPI Base Toolkit

## Key Implementation Details

The GZIP DEFLATE algorithm uses a GZIP-compatible Limpel-Ziv 77 (LZ77) algorithm for data de-duplication and a GZIP-compatible Static Huffman algorithm for bit reduction. The implementation includes three FPGA accelerated tasks (LZ77, Static Huffman, and CRC).

The FPGA implementation of the algorithm enables either one or two independent GZIP compute engines to operate in parallel on the FPGA. The available FPGA resources constrain the number of engines. By default, the design is parameterized to create a single engine when the design is compiled to target Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA). Two engines are created when compiling for Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX), which is a larger device.

This reference design contains two variants: "High Bandwidth" and "Low-Latency."

- The High Bandwidth variant maximizes system throughput without regard for latency. It transfers input/output SYCL Buffers to FPGA-attached DDR. The kernel then operates on these buffers.
- The Low-Latency variant takes advantage of Universal Shared Memory (USM) to avoid these copy operations, allowing the GZIP engine to access input/output buffers in host-memory directly. This reduces latency, but throughput is also reduced. "Latency" in this context is defined as the duration of time between when the input buffer is available in host memory to when the output buffer (i.e., the compressed result) is available in host memory.
The Low-Latency variant is only supported on Intel Stratix® 10 SX.

| Kernel          | Description
|:---             |:---
| LZ Reduction    | Implements an LZ77 algorithm for data de-duplication. The algorithm produces distance and length information that is compatible with the GZIP DEFLATE implementation.
| Static Huffman  | Uses the same Static Huffman codes used by GZIP's DEFLATE algorithm when it chooses a Static Huffman coding scheme for bit reduction. This choice maintains compatibility with GUNZIP.
| CRC             | Adds a CRC checksum based on the input file; the gzip file format requires this

To optimize performance, GZIP leverages techniques discussed in the following FPGA tutorials:
* **Double Buffering to Overlap Kernel Execution with Buffer Transfers and Host Processing** (double_buffering)
* **On-Chip Memory Attributes** (mem_config)

### Source Code

| File                 | Description
|:---                  |:---
| `gzip.cpp`           | Contains the `main()` function and the top-level interfaces to the SYCL* GZIP functions.
| `gzip_ll.cpp`        | Low latency variant of the top level file.
| `gzipkernel.cpp`     | Contains the SYCL* kernels used to implement GZIP.
| `gzipkernel_ll.cpp`  | Low-latency variant of kernels.
| `CompareGzip.cpp`    | Contains code to compare a GZIP-compatible file with the original input.
| `WriteGzip.cpp`      | Contains code to write a GZIP compatible file.
| `crc32.cpp`          | Contains code to calculate a 32-bit CRC compatible with the GZIP file format and to combine multiple 32-bit CRC values. It is only used to account for the CRC of the last few bytes in the file, which are not processed by the accelerated CRC kernel.
| `kernels.hpp`        | Contains miscellaneous defines and structure definitions required by the LZReduction and Static Huffman kernels.
| `crc32.hpp`          | Header file for `crc32.cpp`.
| `gzipkernel.hpp`     | Header file for `gzipkernels.cpp`.
| `gzipkernel)ll.hpp`  | Header file for `gzipkernels_ll.cpp`.
| `CompareGzip.hpp`    | Header file for `CompareGzip.cpp`.
| `pipe_utils.hpp`     | Header file containing the definition of an array of pipes. This header can be found in the `../include/` directory of FPGA section of the repository.
| `WriteGzip.hpp`      | Header file for `WriteGzip.cpp`.

### Compiler Flags Used

| Flag                      | Description
|:---                       |:---
| `-Xshardware`             | Targets FPGA hardware (instead of FPGA emulator).
| `-Xsparallel=2`           | Uses two cores when compiling the bitstream through Intel® Quartus®.
| `-Xsseed=<seed_num>`      | Uses a particular seed while running Intel® Quartus®, selected to yield the best Fmax for this design.
| `-Xsnum-reorder=6`        | On Intel Stratix® 10 SX only, specify a wider data path for read data from global memory.
| `-Xsopt-arg="-nocaching"` | Specifies that cached LSUs should not be used.
| `-DNUM_ENGINES=<1\|2>`    | Specifies that 1 GZIP engine should be compiled when targeting Intel Arria® 10 GX and two engines when targeting Intel Stratix® 10 SX.

### Performance

See the [Performance Disclaimers](#performance-disclaimers) at the end of this README.

| Device                                          | Throughput
|:---                                             |:---
| Intel® PAC with Intel® Arria® 10 GX FPGA        | 1 engine @ 3.4 GB/s
| Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)  | 2 engines @ 5.5 GB/s each = 11.0 GB/s total (High Bandwidth variant) using 120MB+ input <br> 2 engines @ 3.5 GB/s = 7.0 GB/s (Low Latency variant) using 80 KB input

### Additional Documentation
- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `GZIP` Design

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*


1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   For the **low latency** version of the design, add `-DLOW_LATENCY=1`.
   ```
   cmake .. -DLOW_LATENCY=1 -DFPGA_DEVICE=intel_s10sx_pac:pac_s10_usm
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
       ```
       make fpga_emu
       ```
   2. Compile for simulation (medium compile time, targets simulated FPGA device):

      ```
      make fpga_sim
      ```
   3. Generate the HTML performance report.
       ```
       make report
       ```
       The report resides at `gzip_report.prj/reports/report/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
       ```
       make fpga
       ```
   (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/gzip.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/gzip.fpga.tar.gz).

### On Windows*

> **Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   For the **low latency** version of the design, add `-DLOW_LATENCY=1`.
   ```
   cmake -G "Nmake Makefiles" .. -DLOW_LATENCY=1 -DFPGA_DEVICE=intel_s10sx_pac:pac_s10_usm
   ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Compile for simulation (medium compile time, targets simulated FPGA device).
      ```
      nmake fpga_sim
      ```
   3. Generate the HTML performance report.
      ```
      nmake report
      ```
      The report resides at `gzip_report.a.prj/reports/report/report.html`.
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `GZIP` Program

### Configurable Parameters

| Argument             | Description
|:---                  |:---
| `<input_file>`       | Specifies the file to be compressed. <br> Use an 120+ MB file to achieve peak performance. <br> Use an 80 KB file for Low Latency variant.
| `-o=<output_file>`   | Specifies the name of the output file. The default name of the output file is `<input_file>.gz`. <br> When targeting Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), the single `<input_file>` is fed to both engines, yielding two identical output files, using `<output_file>` as the basis for the filenames.

### On Linux

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    ./gzip.fpga_emu <input_file> -o=<output_file>
    ```
    
 2. Run the sample on the FPGA simulator.
    ```
    ./gzip.fpga_sim <input_file> -o=<output_file>
    ```

 3. Run the sample on the FPGA device.
   ```
   aocl initialize acl0 pac_s10_usm
   ./gzip.fpga <input_file> -o=<output_file>
   ```
### On Windows

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    gzip.fpga_emu.exe <input_file> -o=<output_file>
    ```
 2. Run the sample on the FPGA simulator.
    ```
    gzip.fpga_sim.exe <input_file> -o=<output_file>
    ```
 3. Run the sample on the FPGA device.
    ```
    aocl initialize acl0 pac_s10_usm
    gzip.fpga.exe <input_file> -o=<output_file>
    ```

### Build and Run the Samples on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

Use the Linux instructions to build and run the program.

You can specify an FPGA runtime node using a single line script similar to the following example.

```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:fpga_runtime:ppn=2` (lower case L) assigns one full node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes           |Command Options
  |:---                      |:---
  |FPGA Compile Time         |`qsub -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime (Arria 10)   |`qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d .`
  |FPGA Runtime (Stratix 10) |`qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d .`
  |GPU                       |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	                     |`qsub -l nodes=1:xeon:ppn=2 -d .`

>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **24 hours**.

 ### Application Parameters
=======
Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.

## Example Output

```
Running on device:  pac_a10 : Intel PAC Platform (pac_ee00000)
Throughput: 3.4321 GB/s
Compression Ratio 33.2737%
PASSED
```

### Performance Disclaimers

Tests document the performance of components on a particular test on a specific system. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).

Performance results are based on testing as of October 27, 2020 (using tool version 2021.1), and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

Intel measured the performance on October 27, 2020 (using tool version 2021.1).

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

© Intel Corporation.

###  Additional References

- [Gzip Compression OpenCL™ Design Example](https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/gzip-compression.html)
- [RFC 1951 - DEFLATE Compressed Data Format Specification](https://www.ietf.org/rfc/rfc1951.txt)
- [RFC 1952 - GZIP file format specification](https://www.ietf.org/rfc/rfc1952.txt)
- [OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer) GitHub repository

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
