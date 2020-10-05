# GZIP Compression
Reference design demonstrating high-performance GZIP compression on FPGA.
 
***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. Additional reference material specific to this GZIP implementation is provided in the References section of this README.
 
| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC)  with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC)  with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | How to implement a high performance multi-engine compression algorithm on FPGA
| Time to complete                  | 1 hr (not including compile time)
 
_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_
 
**Performance**
Please refer to performance disclaimer at the end of this README.

| Device                                                | Throughput
|:---                                                   |:---
| Intel® PAC with Intel Arria® 10 GX FPGA               | 1 engine @ 3.4 GB/s
| Intel® PAC with Intel Stratix® 10 SX FPGA             | 2 engines @ 5.5 GB/s each = 11.0 GB/s total
 
## Purpose

This DPC++ reference design implements a compression algorithm. The implementation is optimized for the FPGA device. The compression result is GZIP-compatible and can be decompressed with GUNZIP. The GZIP output file format is compatible with GZIP's DEFLATE algorithm, and follows a fixed subset of [RFC 1951](https://www.ietf.org/rfc/rfc1951.txt). See the References section for more specific references. 

The algorithm uses a GZIP-compatible Limpel-Ziv 77 (LZ77) algorithm for data de-duplication, and a GZIP-compatible Static Huffman algorithm for bit reduction. The implementation includes three FPGA accelerated tasks (LZ77, Static Huffman and CRC). 

The FPGA implementation of the algorithm enables either one or two independent GZIP compute engines to operate in parallel on the FPGA. The number of engines is constrained by the available FPGA resources. By default, the design is parameterized to create a single engine when the design is compiled targeting Intel® PAC with Intel Arria® 10 GX FPGA. Two engines are created when targeting Intel® PAC with Intel Stratix® 10 SX FPGA, a larger device.
 
## Key Implementation Details

 | Kernel                     | Description
---                          |---
| LZ Reduction               | Implements a LZ77 algorithm for data de-duplication. The algorithm produces distance and length information that is compatible with GZIP's DEFLATE implementation. 
| Static Huffman             | Uses the same Static Huffman codes used by GZIP's DEFLATE algorithm when it chooses a Static Huffman coding scheme for bit reduction. This choice maintains compatibility with GUNZIP. 
| CRC                        | Adds a CRC checksum based on the input file; this is required by the gzip file format 

To optimize performance, GZIP leverages techniques discussed in the following FPGA tutorials: 
* **Double Buffering to Overlap Kernel Execution with Buffer Transfers and Host Processing** (double_buffering)
* **On-Chip Memory Attributes** (mem_config)


## License  
This code sample is licensed under MIT license.
 
 
## Building the `gzip` Reference Design
 
### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.
 
### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile or fpga_runtime) as well as whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/get-started/base-toolkit/](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)).
 
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
   Alternatively, to compile for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:
 
   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10_usm
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
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/gzip.fpga.tar.gz" download>here</a>.
 
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
     If you are targeting Intel® PAC with Intel Stratix® 10 SX FPGA, instead use:
     ```
     ninja report_s10_pac
     ```     
   * Compiling for FPGA hardware is not yet supported on Windows.
 
 ### In Third-Party Integrated Development Environments (IDEs)
 
You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
 
 
## Running the Reference Design
 
 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./gzip.fpga_emu <input_file> [-o=<output_file>]     (Linux)
     gzip.fpga_emu.exe <input_file> [-o=<output_file>]   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     aocl initialize acl0 pac_s10_usm
     ./gzip.fpga <input_file> [-o=<output_file>]         (Linux)
     ```
 ### Application Parameters

| Argument | Description
---        |---
| `<input_file>` | Mandatory argument that specifies the file to be compressed. Use a 120+ MB file to achieve peak performance.
| `-o=<output_file>` | Optional argument that specifies the name of the output file. The default name of the output file is `<input_file>.gz`. When targeting Intel Stratix® 10 SX, the single `<input_file>` is fed to both engines, yielding two identical output files, using `<output_file>` as the basis for the filenames.
 
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
| `gzipkernel.cpp`            | Contains the SYCL* kernels used to implement GZIP. 
| `CompareGzip.cpp`            | Contains code to compare a GZIP-compatible file with the original input.
| `WriteGzip.cpp`              | Contains code to write a GZIP compatible file. 
| `crc32.cpp`                  | Contains code to calculate a 32-bit CRC that is compatible with the GZIP file format and to combine multiple 32-bit CRC values. It is used to account only for the CRC of the last few bytes in the file, which are not processed by the accelerated CRC kernel. 
| `kernels.hpp`                  | Contains miscellaneous defines and structure definitions required by the LZReduction and Static Huffman kernels.
| `crc32.hpp`                    | Header file for `crc32.cpp`.
| `gzipkernel.hpp`              | Header file for `gzipkernels.cpp`.
| `CompareGzip.hpp`              | Header file for `CompareGzip.cpp`.
| `WriteGzip.hpp`                | Header file for `WriteGzip.cpp`. 

### Compiler Flags Used

| Flag | Description
---    |---
`-Xshardware` | Target FPGA hardware (as opposed to FPGA emulator)
`-Xsparallel=2` | Uses 2 cores when compiling the bitstream through Quartus
`-Xsseed=19` | Uses seed 19 during Quartus, yields slightly higher fmax
`-Xsnum-reorder=6` | On Intel Stratix® 10 SX only, specify a wider data path for read data from global memory 
`-DNUM_ENGINES=<1|2>` | Specifies that 1 GZIP engine should be compiled when targeting Arria® 10 GX and 2 engines when targeting Intel Stratix® 10 SX


### Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of July 29, 2020 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 29, 2020

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.

### References
[Khronous SYCL Resources](https://www.khronos.org/sycl/resources)

[Intel GZIP OpenCL Design Example](https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/gzip-compression.html)

[RFC 1951 - DEFLATE Data Format](https://www.ietf.org/rfc/rfc1951.txt)

[RFC 1952 - GZIP Specification 4.3](https://www.ietf.org/rfc/rfc1952.txt)

[OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer)

