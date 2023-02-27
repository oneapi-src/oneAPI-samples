# Read-Only Cache
This FPGA tutorial demonstrates how to use the read-only cache feature to boost
the throughput of a SYCL*-compliant FPGA design that requires reading from off-chip
memory in a non-contiguous manner.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Agilex®, Arria® 10, and Stratix® 10 FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | How and when to use the read-only cache feature
| Time to complete                  | 30 minutes

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

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")
   
   tier1 --> tier2 --> tier3 --> tier4
   
   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

## Purpose

This FPGA tutorial demonstrates an example of using the read-only cache to
boost the throughput of a FPGA design. Specifically, the read-only cache
is most appropriate for non-contiguous read accesses from off-chip memory
buffers that are guaranteed to be constant throughout the execution of a
kernel. The read-only cache is optimized for high cache hit performance.

To enable the read-only cache, the `-Xsread-only-cache-size<N>` flag should be
passed to the `icpx` command. Each kernel will get its own *private* version
of the cache that serves all reads in the kernel from read-only no-alias
accessors. Read-only no-alias accessors are accessors that have both the
`read_only` and the `no_alias` properties:
```c++
accessor sqrt_lut(sqrt_lut_buf, h, read_only, accessor_property_list{no_alias});
```
The `read_only` property is required because the cache is *read_only*. The
`no_alias` property is required to guarantee that the buffer will not be
written to from the kernel through another accessor or through a USM pointer.
You do not need to apply the `no_alias` property if the kernel already has the
`[[intel::kernel_args_restrict]]` attribute.

Each private cache is also replicated as many times as needed so that it can
expose extra read ports. The size of each replicate is `N` bytes as specified
by the `-Xsread-only-cache-size=<N>` flag.

### Tutorial Design
The basic function performed by the tutorial kernel is a series of table
lookups from a buffer (`sqrt_lut_buf`) that contains the square root values of
the first 512 integers. By default, the compiler will generate load-store units
(LSUs) that are optimized for the case where global memory accesses are
contiguous. When the memory accesses are non-contiguous, as is the case in this
tutorial design, these LSUs tend to suffer major throughput loss. The read-only
cache can sometimes help in such situations, especially when sized correctly.

This tutorial requires compiling the source code twice: once with the
`-Xsread-only-cache-size=<N>` flag and once without it. Because the look-up
table contains 512 integers as indicated by the `kLUTSize` constant, the chosen
size of the cache is `512*4 bytes = 2048 bytes`, and so, the flag
`-Xsread-only-cache-size=2048` is passed to `icpx`.

## Key Concepts
* How to use the read-only cache feature
* The scenarios in which this feature can help improve the throughput of a
  design

## Building the `read_only_cache` Tutorial

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

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex® device family), run `cmake` using the command:
  ```
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
   * Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
     ```
     make fpga_sim
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex® device family), run `cmake` using the command:
  ```
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
   * Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
     ```
     nmake fpga_sim
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

> **Note**: If you encounter any issues with long paths when compiling under
Windows*, you may have to create your ‘build’ directory in a shorter path, for
example c:\samples\build.  You can then run cmake from that directory, and
provide cmake with the full path to your sample directory.

## Examining the Reports
Locate the pair of `report.html` files in the
`read_only_cache_disabled_report.prj` and `read_only_cache_enabled_report.prj`
directories. Open the reports in Chrome*, Firefox*, Edge*, or Internet
Explorer*. Navigate to the "Area Analysis of System" section of each report
(Area Analysis > Area Analysis of System) and expand the "Kernel System" entry
in the table. Notice that when the read-only cache is enabled, a new entry
shows up in the table called "Constant cache interconnect" indicating that the
cache has been created.


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./read_only_cache.fpga_emu     (Linux)
     read_only_cache.fpga_emu.exe   (Windows)
     ```
    Note that the read-only cache is not implemented in emulation. The
    `-Xsread-only-cache-size<N>` flag does not impact the emulator in any way
    which is why we only have a single executable for this flow.
2. Run the sample on the FPGA simulation device (two executables should be generated):
  * On Linux
    ```bash
    CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./read_only_cache.fpga_sim
    ```
  * On Windows
    ```bash
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
    read_only_cache.fpga_sim.exe
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
    ```
    
    Note although the circuit for the read-only cache is implemented in 
    simulation, one cannot see consistent performance increase with the cache 
    enabled as each clock cycle in the simulator doesn't have a consistent 
    latency, as it does in the hardware. For this reason there is just a single
    executable for this flow.
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`):
     ```
     ./read_only_cache_disabled.fpga         (Linux)
     ./read_only_cache_enabled.fpga          (Linux)
     read_only_cache_disabled.fpga.exe       (Windows)
     read_only_cache_enabled.fpga.exe        (Windows)
     ```

### Example of Output

Running `./read_only_cache_disabled.fpga`:
```

SQRT LUT size: 512
Number of outputs: 131072
Verification PASSED

Kernel execution time: 0.003377 seconds
Kernel throughput: 148.06 MB/s
```

Running `./read_only_cache_disabled.fpga`:
```

SQRT LUT size: 512
Number of outputs: 131072
Verification PASSED

Kernel execution time: 0.001675 seconds
Kernel throughput with the read-only cache: 298.51 MB/s
```

### Discussion of Results

A test compile of this tutorial design achieved the following results on the
Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA:

Configuration     | Execution Time (ms) | Throughput (MB/s)
|:---             |:---                 |:---
|Without caching  | 3.377               | 148.06
|With caching     | 1.675               | 298.51

When the read-only cache is enabled, performance notably increases. As
previously mentioned, when the global memory accesses are random (i.e.
non-contiguous), enabling the read-only cache and sizing it correctly may allow
the design to achieve a higher throughput.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
