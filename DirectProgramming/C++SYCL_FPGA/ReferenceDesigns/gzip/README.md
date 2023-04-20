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

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

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

## Key Implementation Details

The GZIP DEFLATE algorithm uses a GZIP-compatible Limpel-Ziv 77 (LZ77) algorithm for data de-duplication and a GZIP-compatible Static Huffman algorithm for bit reduction. The implementation includes three FPGA accelerated tasks (LZ77, Static Huffman, and CRC).

The FPGA implementation of the algorithm enables either one or two independent GZIP compute engines to operate in parallel on the FPGA. The available FPGA resources constrain the number of engines. By default, the design is parameterized to create a single engine when the design is compiled to target an Intel® Arria® 10 FPGA. Two engines are created when compiling for Intel® Stratix® 10 or Agilex® 7 FPGAs, which are a larger device.

This reference design contains two variants: "High Bandwidth" and "Low-Latency."

- The High Bandwidth variant maximizes system throughput without regard for latency. It transfers input/output SYCL Buffers to FPGA-attached DDR. The kernel then operates on these buffers.
- The Low-Latency variant takes advantage of Universal Shared Memory (USM) to avoid these copy operations, allowing the GZIP engine to access input/output buffers in host-memory directly. This reduces latency, but throughput is also reduced. "Latency" in this context is defined as the duration of time between when the input buffer is available in host memory to when the output buffer (i.e., the compressed result) is available in host memory.
The Low-Latency variant is only supported on USM capable BSPs, or when targeting an FPGA family/part number.

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
| `-Xsnum-reorder=6`        | On FPGA boards that have a large memory bandwidth, specify a wider data path for read data from global memory.
| `-Xsopt-arg="-nocaching"` | Specifies that cached LSUs should not be used.

Additionaly, the cmake build system can be configured using the following parameter:

| cmake option              | Description
|:---                       |:---
| `-DNUM_ENGINES=<1\|2>`    | Specifies that the number of GZIP engine that should be compiled.

### Performance

Performance results are based on testing as of October 27, 2020.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                                                              | Throughput
|:---                                                                                 |:---
| Intel® PAC with Intel® Arria® 10 GX FPGA                                            | 1 engine @ 3.4 GB/s
| Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)  | 2 engines @ 4.5 GB/s each = 9.0 GB/s total (High Bandwidth variant) using 120MB+ input <br> 2 engines @ 3.5 GB/s = 7.0 GB/s (Low Latency variant) using 80 KB input

## Build the `GZIP` Design

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

   For the **low latency** version of the design, add `-DLOW_LATENCY=1` to your `cmake` command.

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

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

### On Windows*

1. Change to the sample directory.
2. Configure the build system for the Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```

   For the **low latency** version of the design, add `-DLOW_LATENCY=1` to your `cmake` command.

  > **Note**: You can change the default target by using the command:
  >  ```
  >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
  >  ```
  >
  > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
  >  ```
  >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
  >  ```
  >
  > You will only be able to run an executable on the FPGA if you specified a BSP.

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

## Run the `GZIP` Program

### Configurable Parameters

| Argument             | Description
|:---                  |:---
| `<input_file>`       | Specifies the file to be compressed. <br> Use an 120+ MB file to achieve peak performance. <br> Use an 80 KB file for Low Latency variant. <br> Use a smaller file such as an 100 B file if the simulator flow is taking too long.
| `-o=<output_file>`   | Specifies the name of the output file. The default name of the output file is `<input_file>.gz`. <br> When using two engines, the single `<input_file>` is fed to both engines, yielding two identical output files, using `<output_file>` as the basis for the filenames.

### On Linux

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    ./gzip.fpga_emu <input_file> -o=<output_file>
    ```

 2. Run the sample on the FPGA simulator.
    ```
    CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./gzip.fpga_sim <input_file> -o=<output_file>
    ```
    For the smaller file option.
    ```
    CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./gzip.fpga_sim ../data/100b.txt -o=<output_file>
    ```
 3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
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
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
    gzip.fpga_sim.exe <input_file> -o=<output_file>
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
    ```
    For the smaller file option.
    ```
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
    gzip.fpga_sim.exe ../data/100b.txt -o=<output_file>
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
    ```
 3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
    ```
    aocl initialize acl0 pac_s10_usm
    gzip.fpga.exe <input_file> -o=<output_file>
    ```

## Example Output

```
Running on device:  pac_a10 : Intel PAC Platform (pac_ee00000)
Throughput: 3.4321 GB/s
Compression Ratio 33.2737%
PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
