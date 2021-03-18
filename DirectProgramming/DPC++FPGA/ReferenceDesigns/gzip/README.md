# GZIP Compression
Reference design demonstrating high-performance GZIP compression on FPGA.
 
***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. <br>
Additional reference material specific to this GZIP implementation is provided in the References section of this README.

 
| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | How to implement a high-performance multi-engine compression algorithm on FPGA
| Time to complete                  | 1 hr (not including compile time)
 
 
***Performance***: Please refer to the performance disclaimer at the end of this README.

| Device                                                | Throughput
|:---                                                   |:---
| Intel® PAC with Intel Arria® 10 GX FPGA               | 1 engine @ 3.4 GB/s
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)             | 2 engines @ 5.5 GB/s each = 11.0 GB/s total (High Bandwidth variant) using 120MB+ input, 2 engines @ 3.5 GB/s = 7.0 GB/s (Low Latency variant) using 80kB input

 
## Purpose

This DPC++ reference design implements a compression algorithm. The implementation is optimized for the FPGA device. The compression result is GZIP-compatible and can be decompressed with GUNZIP. The GZIP output file format is compatible with GZIP's DEFLATE algorithm and follows a fixed subset [RFC 1951](https://www.ietf.org/rfc/rfc1951.txt). See the References section for more specific references. 

The algorithm uses a GZIP-compatible Limpel-Ziv 77 (LZ77) algorithm for data de-duplication and a GZIP-compatible Static Huffman algorithm for bit reduction. The implementation includes three FPGA accelerated tasks (LZ77, Static Huffman and CRC). 

The FPGA implementation of the algorithm enables either one or two independent GZIP compute engines to operate in parallel on the FPGA. The available FPGA resources constrain the number of engines. By default, the design is parameterized to create a single engine when the design is compiled to target Intel® PAC with Intel Arria® 10 GX FPGA. Two engines are created when compiling for Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), a larger device.

This reference design contains two variants: "High Bandwidth" and "Low-Latency."
The High Bandwidth variant maximizes system throughput without regard for latency. It transfers input/output SYCL Buffers to FPGA-attached DDR. The kernel then operates on these buffers.
The Low-Latency variant takes advantage of Universal Shared Memory (USM) to avoid these copy operations, allowing the GZIP engine to directly access input/output buffers in host-memory. This reduces latency, but throughput is also reduced. "Latency" in this context is defined as the duration of time between when the input buffer is available in host memory to when the output buffer (i.e., the compressed result) is available in host memory.
The Low-Latency variant is only supported on Stratix® 10 SX.
 
## Key Implementation Details

 | Kernel                     | Description
---                          |---
| LZ Reduction               | Implements an LZ77 algorithm for data de-duplication. The algorithm produces distance and length information that is compatible with GZIP's DEFLATE implementation. 
| Static Huffman             | Uses the same Static Huffman codes used by GZIP's DEFLATE algorithm when it chooses a Static Huffman coding scheme for bit reduction. This choice maintains compatibility with GUNZIP. 
| CRC                        | Adds a CRC checksum based on the input file; the gzip file format requires this 

To optimize performance, GZIP leverages techniques discussed in the following FPGA tutorials: 
* **Double Buffering to Overlap Kernel Execution with Buffer Transfers and Host Processing** (double_buffering)
* **On-Chip Memory Attributes** (mem_config)


## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
 
 
## Building the `gzip` Reference Design
 
### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.
 
### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).
 
When compiling for FPGA hardware, it is recommended to increase the job timeout to 24h.
 
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
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10_usm
   ```
2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:
 
   * Compile for emulation (fast compile time, targets emulated FPGA device): 
      ```
      make fpga_emu
      ```
    > Note: for the Low Latency variant, use `make fpga_emu_ll`. Only supported on Stratix® 10 SX.

   * Generate the optimization report: 
     ```
     make report
     ``` 
    > Note: for the Low Latency variant, use `make report_ll`. Only supported on Stratix® 10 SX.

   * Compile for FPGA hardware (longer compile time, targets FPGA device): 
     ```
     make fpga
     ``` 
    > Note: for the Low Latency variant, use `make fpga_ll`. Only supported on Stratix® 10 SX.
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/gzip.fpga.tar.gz" download>here</a>.
 
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
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10_usm
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
     ```
     nmake fpga_emu
     ```
    > Note: for the Low Latency variant, use `nmake fpga_emu_ll`. Only supported on Stratix® 10 SX.
   * Generate the optimization report: 
     ```
     nmake report
     ``` 
    > Note: for the Low Latency variant, use `nmake report_ll`. Only supported on Stratix® 10 SX.   
   * An FPGA hardware target is not provided on Windows*. 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.
 
 ### In Third-Party Integrated Development Environments (IDEs)
 
You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
 
 
## Running the Reference Design
 
 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./gzip.fpga_emu <input_file> [-o=<output_file>]     (Linux)
     gzip.fpga_emu.exe <input_file> [-o=<output_file>]   (Windows)
     ```
    > Note: for the Low Latency variant use `gzip_ll.fpga_emu`. Only supported on Stratix® 10 SX.
2. Run the sample on the FPGA device:
     ```
     aocl initialize acl0 pac_s10_usm
     ./gzip.fpga <input_file> [-o=<output_file>]         (Linux)
     ```
     > Note: for the Low Latency variant use `gzip_ll.fpga`. Only supported on Stratix® 10 SX.
 ### Application Parameters

| Argument | Description
---        |---
| `<input_file>` | Mandatory argument that specifies the file to be compressed. Use a 120+ MB file to achieve peak performance (80kB for Low Latency variant).
| `-o=<output_file>` | Optional argument that specifies the name of the output file. The default name of the output file is `<input_file>.gz`. When targeting Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), the single `<input_file>` is fed to both engines, yielding two identical output files, using `<output_file>` as the basis for the filenames.
 
### Example of Output
 
```
Running on device:  pac_a10 : Intel PAC Platform (pac_ee00000)
Throughput: 3.4321 GB/s
Compression Ratio 33.2737%
PASSED
```
## Additional Design Information
### Source Code Explanation

| File                         | Description 
---                            |---
| `gzip.cpp`                   | Contains the `main()` function and the top-level interfaces to the SYCL* GZIP functions.
| `gzip_ll.cpp`                | Low latency variant of the top level file.
| `gzipkernel.cpp`             | Contains the SYCL* kernels used to implement GZIP. 
| `gzipkernel_ll.cpp`          | Low-latency variant of kernels.
| `CompareGzip.cpp`            | Contains code to compare a GZIP-compatible file with the original input.
| `WriteGzip.cpp`              | Contains code to write a GZIP compatible file. 
| `crc32.cpp`                  | Contains code to calculate a 32-bit CRC compatible with the GZIP file format and to combine multiple 32-bit CRC values. It is only used to account for the CRC of the last few bytes in the file, which are not processed by the accelerated CRC kernel. 
| `kernels.hpp`                  | Contains miscellaneous defines and structure definitions required by the LZReduction and Static Huffman kernels.
| `crc32.hpp`                    | Header file for `crc32.cpp`.
| `gzipkernel.hpp`              | Header file for `gzipkernels.cpp`.
| `gzipkernel)ll.hpp`              | Header file for `gzipkernels_ll.cpp`.
| `CompareGzip.hpp`              | Header file for `CompareGzip.cpp`.
| `pipe_array.hpp`                | Header file containing the definition of an array of pipes. 
| `pipe_array_internal.hpp`       | Helper for pipe_array.hpp. 
| `WriteGzip.hpp`                | Header file for `WriteGzip.cpp`. 

### Compiler Flags Used

| Flag | Description
---    |---
`-Xshardware` | Target FPGA hardware (as opposed to FPGA emulator)
`-Xsparallel=2` | Uses two cores when compiling the bitstream through Quartus
`-Xsseed=8` | Uses seed 8 (seed 33 for Low latency Variant) during Quartus, yields slightly higher fmax
`-Xsnum-reorder=6` | On Intel Stratix® 10 SX only, specify a wider data path for read data from global memory 
`-Xsopt-arg="-nocaching"` | Specifies that cached LSUs should not be used.
`-DNUM_ENGINES=<1|2>` | Specifies that 1 GZIP engine should be compiled when targeting Intel Arria® 10 GX and two engines when targeting Intel Stratix® 10 SX


### Performance disclaimers

Tests document the performance of components on a particular test on a specific system. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of October 27, 2020 (using tool version 2021.1), and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

Intel measured the performance on October 27, 2020 (using tool version 2021.1).

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.

### References
[Khronos SYCL Resources](https://www.khronos.org/sycl/resources)

[Intel GZIP OpenCL Design Example](https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/gzip-compression.html)

[RFC 1951 - DEFLATE Data Format](https://www.ietf.org/rfc/rfc1951.txt)

[RFC 1952 - GZIP Specification 4.3](https://www.ietf.org/rfc/rfc1952.txt)

[OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer)

