# Read-Only Cache 
This FPGA tutorial demonstrates how to use the read-only cache feature to boost
the throughput of an FPGA DPC++ design that uses a read-only look-up table
defined in host code.

***Documentation***:  The [DPC++ FPGA Code Samples
Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html)
helps you to navigate the samples and build your knowledge of DPC++ for FPGA.
<br>
The [oneAPI DPC++ FPGA Optimization
Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)
is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming
Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general
resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | How and when to use the read-only cache feature
| Time to complete                  | 30 minutes



## Purpose

This FPGA tutorial demonstrates an example of using the read-only cache to
boost the throughput of a oneAPI FPGA design. Specifically, the read-only cache
is most appropriate for table lookups that are constant throughput the
execution of a kernel and is optimized for high cache hit performance.

To enabled the read-only cache, the `-Xsread-only-cache-size<N>` flag should be
passed to the `dpcpp` command. Each kernel will gets its own version of the
cache that serves all reads in the kernel from read-only no-alias accessors.
Each private cache is also replicated as many times as needed to expose extra
read ports. The size of each replicate is `<N>` bytes as specified by the
`-Xsread-only-cache-size=<N>` flag.

### Understanding the Tutorial Design

The basic function performed by the tutorial kernel is a computing and
accumulating the square root of a large number of randomly generated integers
using a look-up table generated on the host. 

#### Part 1: Without the `-Xsread-only-cache-size=<N>` flag

...

#### Part 2: With the `-Xsread-only-cache-size=<N>` flag

...

## Key Concepts
* How to use the read-only cache feature 
* The scenarios in which this feature can help improve the throughput of a
  design 

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `read_only_cache` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at
`%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the
type of compute node and whether to run in batch or interactive mode. Compiles
to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA
hardware is only supported on fpga_runtime nodes of the appropriate type, such
as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor
executing programs on FPGA hardware are supported on the login nodes. For more
information, see the Intel® oneAPI Base Toolkit Get Started Guide
([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout
to 12h.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake`
   using the command:  
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix®
   10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board
   support package is installed on your system. Then run `cmake` using the
   command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build
   targets are provided, matching the recommended development flow:

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
3. (Optional) As the above hardware compile may take several hours to complete,
   FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be
   downloaded <a
   href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/read_only_cache.fpga.tar.gz"
   download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake`
   using the command:  
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix®
   10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board
   support package is installed on your system. Then run `cmake` using the
   command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build
   targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report: 
     ```
     nmake report
     ``` 
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ``` 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005
(with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware
on Windows* requires a third-party or custom Board Support Package (BSP) with
Windows* support.
 
### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the
Visual Studio* IDE (in Windows*). For instructions, refer to the following
link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party
IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)


## Examining the Reports
Locate the pair of `report.html` files in either:

* **Report-only compile**: `read_only_cache_disabled_report.prj` and
  `read_only_cache_enabled_report.prj`
* **FPGA hardware compile**: `read_only_cache_disabled.prj` and
  `read_only_cache_enabled.prj`

Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.
Notice that ...

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./read_only_cache.fpga_emu     (Linux)
     read_only_cache.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./read_only_cache_disabled.fpga         (Linux)
     ./read_only_cache_enabled.fpga         (Linux)
     ```

### Example of Output

```
Platform name: Intel(R) FPGA SDK for OpenCL(TM)
Device name: pac_a10 : Intel PAC Platform (pac_ee00000)


SQRT LUT size: 16777216
Number of outputs: 64

Verification PASSED

Kernel execution time: 0.114106 seconds
Kernel throughput: 560.884047 MB/s
```

### Discussion of Results

A test compile of this tutorial design achieved an f<sub>MAX</sub> of
approximately <fmax> MHz on the Intel® Programmable Acceleration Card with
Intel® Arria® 10 GX FPGA. The results are shown in the following table:

Configuration | Execution Time (ms) | Throughput (MB/s)
-|-|-
Without caching | <t> | <t>
With caching | <t> | <t>

When caching is used, performance notably increases. As previously mentioned, 
