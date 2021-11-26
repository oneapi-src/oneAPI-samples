# Memory Channels 
This FPGA tutorial demonstrates how to use the `mem_channel` buffer property in
conjuction with the `-Xsno-interleaving` flag to reduce the area consumed by a
DPC++ FPGA design.

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
| What you will learn               | How and when to use the `mem_channel` buffer property and the `-Xsno-interleaving` flag
| Time to complete                  | 30 minutes



## Purpose

This FPGA tutorial demonstrates an example of using the `mem_channel` buffer
property in conjuction with the `-Xsno-interleaving` flag to reduce the amount
of resources required to implement a DPC++ FPGA design.

By default, the Intel® oneAPI DPC++ compiler configures each global memory type
in a burst-interleaved manner where memory words are interleaved across the
available memory channels. This usually leads to better throughput because it
prevents load imbalance by ensuring that memory accesses to not favor one
external memory channel over another. However, this configuration can be
expensive in terms of FPGA resources because the global memory interconnect
required to orchestrate the memory accesses across all the channels is complex. 

The Intel® oneAPI DPC++ compiler allows to avoid this area overhead by
disabling burst-interleaving and assigning buffers to invidual channels. There
are two advantages for such configuration:
1. A simpler global memory interconnect is built which requires a smaller
   amount of FPGA resources than the interconnect needed for the
   burst-interleaving configuration.
2. Potential improvements to the global memory bandwidth utilization due to
   less contention at each memory channel.

Burst-interleaving should only be disabled in situations where satisfactory
load balancing can be achived by assigning buffers to individual channels.
Otherwise, the global memory bandwidth utilization may be reduced down which
will negatively impact the throughput of your design. 

To disable burst-interleaving, you need to assign a memory channel to each
buffer using the `mem_channel` buffer property:
```c++
buffer a_buf(a_vec, {property::buffer::mem_channel{1}});
buffer b_buf(b_vec, {property::buffer::mem_channel{2}});
```
The ID of the lowest available memory channel is 1. You also need to pass the
`-Xsno-interleaving` to your `dpcpp` command. 

Note that for FPGA boards that have multiple memory types, it is possible to
select which memory you want to disable burst-interleaving for by passing the
memory type to the `-Xsno-interleaving` flag:
`-Xsno-interleaving=<global_memory_type>`. The memory type is usually indicated
in the board specification XML file.


### Tutorial Design 
The basic function performed by the tutorial kernel is an addition of 3
vectors. When burst-interleaving is disabled, each buffer is assigned to a
specific memory channel depending on how many channels are available. 

In the `CMakeLists.txt` file, the macro `NO_INTERLEAVING` is defined when the
`-Xsno-interleaving` flag is passed to the `dpcpp` command. 

The macro `FOUR_CHANNELS` is defined only when the design is compiled for the
Stratix® 10 GX FPGA because that board has an external memory with four
available channels. In that case, each of the 4 buffers in this design is
assigned to one of the available channels. 

When the design is compiled for the Arria® 10 GX FPGA, the 4 buffers are
equally assigned to the those two available channels on that board.


## Key Concepts
* How to use the `mem_channel` buffer property in conjuction with the
  `-Xsno-interleaving` flag.
* The scenarios in which this feature can help reduce the area consumed by a
  DPC++ FPGA design without impacting throughput.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `mem_channels` Tutorial

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
   href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/mem_channels.fpga.tar.gz"
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
Locate the pair of `report.html` files in the `mem_channels_interleaving.prj`
and `mem_channels_no_interleaving.prj` directories. Open the reports in any of
Chrome*, Firefox*, Edge*, or Internet Explorer*. In the "Summary" tab, locate
the "Quartus Fitter Resource Utilization Summary" entry and expand it to see a
table showing the FPGA resources that were allocated for the design. Notice
that when burst-interleaving is disabled, the FPGA resources required are
significantly lower than the case where burst-interleaving is enabled.


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./mem_channels.fpga_emu     (Linux)
     mem_channels.fpga_emu.exe   (Windows)
     ```
    Note that the `mem_channel` property and the `-Xsno-interleaving` flag have
    no impact on the emulator which is why we only have a single executable for
    this flow.
2. Run the sample on the FPGA device (two executables should be generated):
     ```
     ./mem_channels_interleaving.fpga         (Linux)
     ./mem_channels_no_interleaving.fpga         (Linux)
     ```

### Example of Output

Running `./mem_channels_interleaving.fpga`:
```
Vector size: 100000 
Verification PASSED

Kernel execution time: 0.004004 seconds
Kernel throughput 749.230914 MB/s
```

Running `./mem_channels_interleaving.fpga`:
```
Vector size: 100000 
Verification PASSED

Kernel execution time: 0.003767 seconds
Kernel throughput 796.379552 MB/s
```

### Discussion of Results

A test compile of this tutorial design achieved the following results on the
Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The tables
shows the performance of the design as well as the resources consumed by the
kernel system.
Configuration | Execution Time (ms) | Throughput (MB/s) | ALM | REG | MLAB | RAM | DSP
-|-|-|-|-|-|-|-
Without `-Xsno-interleaving` | 4.004 | 749.23 | 23,815.4 | 26,727  | 1094 | 53 | 0 
With `-Xsno-interleaving` | 3.767 | 796.38 | 7,060.7  | 16,396  | 38 | 41  | 0

Similarly, when compiled for the Intel® Programmable Acceleration Card with
Intel® Stratix® 10 SX FPGA, the tutorial design achieved the following results:
Configuration | Execution Time (ms) | Throughput (MB/s) | ALM | REG | MLAB | RAM | DSP
-|-|-|-|-|-|-|-
Without `-Xsno-interleaving` | 2.913  | 1029.90 | 14,999.6 | 47,532 | 11 | 345 | 0 
With `-Xsno-interleaving` | 2.913 | 1029.77 | 9,564.1 | 28,616 | 11 | 186 | 0

Notice that the throughput of the design when burst-interleaving is disabled is
equal or better than when burst-interleaving is enabled. However, the resource
utilization is significantly lower without burst-interleaving. Therefore, this
is a design where disabling burst-interleaving and manually assigning buffers
to channels is a net win.
