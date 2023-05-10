# `LSU Control` Sample

This sample is an FPGA tutorial that demonstrates how to configure the load-store units (LSU) in SYCL*-compliant programs using the LSU controls extension.

| Area                 | Description
|:--                   |:--
| What you will learn  | The basic concepts of LSU styles and LSU modifiers. <br>  How to use the LSU controls extension to request specific configurations. <br>  How to confirm what LSU configurations are implemented. <br> A case study of the type of area trade-offs enabled by LSU.
| Time to complete     | 30 minutes
| Category             | Concepts and Functionality

## Purpose

The compiler creates load-store units (LSU) to access off-chip data. The compiler has many options to choose from when configuring each LSU. The SYCL*-compliant LSU controls extension allows you to override the compiler's internal heuristics and control the architecture of each LSU. An introduction to the extension in this tutorial will explain the available options, extension defaults, appropriate use cases, and area trade-offs.

## Prerequisites

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

> **Warning**: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.

## Key Implementation Details

The sample illustrates the following important concepts.

- The basic concepts of LSU styles and LSU configurability.
- How to use the LSU controls extension to request specific configurations.
- How to confirm what LSU configurations are implemented.
- A case study of the type of area trade-offs enabled by the LSU controls extension.

### LSUs and LSU Styles

An LSU is a block that handles loading and storing data to and from memory. Off-chip memory can have variable latency. To mitigate this, different LSU implementations, referred to as styles, are available.

The two LSU styles used in this tutorial are listed below:

| Burst-Coalesced LSU                                                       | Prefetching LSU
|:---                                                                       |:---
| Dynamically buffers requests until the largest possible burst can be made | Buffers the next contiguous address of data for future loads
| Makes efficient accesses to global memory but takes more FPGA resources   | If the access is not contiguous, the buffer must be flushed
| Works for both loads and stores                                           | Works only for loads

The best LSU style depends on the memory access pattern in your design. There are trade-offs for each LSU, so picking a configuration solely based on the area may compromise throughput. In this tutorial, the access pattern is ideal for the prefetching LSU, and it will achieve the same throughput as the burst coalesced LSU, but this is not always the case.

In addition to these two styles, there are also LSU modifiers. LSU modifiers are add-ons that can be combined with LSU styles, such as caching, which can be combined with the burst-coalesced LSU style.

For more details on LSU modifiers and LSU styles, refer to the Memory Accesses section in the [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide).

### Introduction to the LSU Control Extension

The class: ```ext::intel::lsu``` enables you to control the architecture of the LSU. The class has two member functions, `load()` and `store()`, which allow loading from and storing to a global pointer.

The table below summarizes the LSU control extension parameters. The parameters will be respected to the extent possible.

|Control                              | Value                  | Default  | Supports
|:---                                 |:---                    |:---      |:---
|`ext::intel::burst_coalesce<B>`      | B is a Boolean         | false    | both load & store
|`ext::intel::cache<N>`               | N is an integer >=  0  | 0        | only load
|`ext::intel::statically_coalesce<B>` | B is a Boolean         | true     | both load & store
|`ext::intel::prefetch<B>`            | B is a Boolean         | false    | only load

If the default options are used, a pipelined LSU is implemented.

#### Example: Controlling the `prefetch` and `statically_coalesce` Parameters

```c++
//Creating typedefs using the LSU controls class
//for each combination of LSU options desired.
using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,
                                  ext::intel::statically_coalesce<false>>;
// ...
q.submit([&](handler &h) {
  h.single_task<Kernel>([=] {
    //Pointer to external memory
    auto input_ptr = input_accessor.get_pointer();

    //Compiler will use a Prefetch LSU for this load
    int in_data = PrefetchingLSU::load(input_ptr);

    //...
  });
});
```

Currently, not every combination of parameters is valid in the compiler.
For more details on the descriptions of LSU controls, styles, and modifiers refer to the *FPGA LSU Controls* section in the [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide).

### Tutorial Overview

The compiler selects an LSU configuration based on the design's memory access pattern and dependencies. In some cases, it can be beneficial to set the configuration explicitly using the LSU control extension. If the LSU configurations are invalid, the compiler will emit a warning and choose a valid LSU. This tutorial will highlight a situation where using the extension to specify an LSU is beneficial.

In the tutorial, there are three kernels with the same body:

|Kernel Name        | How it loads from the read accessor
|:---               |:---
| KernelPrefetch    | `ext::intel::lsu<ext::intel::prefetch<true>>`
| KernelBurst       | `ext::intel::lsu<ext::intel::burst_coalesce<true>>`
| KernelDefault     | directly loads data from read accessor, instead of using the `ext::intel::lsu` class

The kernel design requests data from global memory in a contiguous manner. Therefore, both the prefetching LSU and the burst-coalesced LSU would allow the design to have high throughput. However, the prefetching LSU is highly optimized for such access patterns, especially in situations where we know, at compile time, that such access pattern exists. This will generally lead to significant area savings. As a result, between the two kernels, `KernelPrefetch` and `KernelBurst`, an improvement in area should be observed with `KernelPrefetch`. The kernel `KernelDefault` shows the same design without using the LSU controls extension. This kernel acts as both a baseline and illustrates the difference in syntax between using the LSU controls and not using them.

## Build the `LSU Control` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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
2. Build the program for Intel® Agilex® 7 device family, which is the default.
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

   1. Compile and run for emulation (fast compile time, targets emulates an FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the HTML optimization reports. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      make report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile and run on FPGA hardware (longer compile time, targets an FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Intel® Agilex® 7 device family, which is the default.
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
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      nmake report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced problem size).
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Read the Reports

Locate `report.html` in the `lsu_control.prj/reports/` directory.

#### Verify Selected LSU

1. Navigate to Views > System Viewers.
   - In this view, click any Kernel basic block on the left pane. All blue icons in the graph refer to external memory LSUs. Click the icons for more details.
   - The word "LD" denotes a load, and the word "ST" denotes the store. Feel free to explore the graphs.

2. In the Kernel pane, click the label `KernelPrefetch.B1` to see a blue LD icon. Hovering over this icon should indicate that a prefetching LSU was inferred.

3. In the Kernel pane, click the label `KernelBurst.B1` to see a blue LD icon. Hovering over this icon should indicate that a burst-coalesced LSU was inferred.

4. In the Kernel pane, click the label `KernelDefault.B1` to see a blue LD icon. Hovering over this icon indicates the LSU style inferred.

This view provides additional information about the LSU architecture selected in the details pane.

#### Area Analysis

1. Navigate to Area Analysis of System (Area Estimates > Area Analysis of System).
   - In this view, you can see how much of the FPGA resources are being utilized (ALUTs, FFs, RAMs, MLABs, DSPs).
   - A design that takes less area will use less of these resources.

2. In the center pane, expand the heading of Kernel System
   - Three kernels should be listed: `KernelBurst`, `KernelDefault`, `KernelPrefetch`.
   - Examine which kernel uses the most resources.

The `KernelPrefetch` should use fewer FPGA resources, as the design is a memory access pattern that lends itself to that LSU style best.

#### Trade-Offs

There are many different ways to configure an LSU. As a programmer, the implementation you should choose depends on your design constraints.

If your design is limited by the available FPGA resources, you can try the following:

- Request a prefetching LSU instead of a burst-coalesced LSU for loads.
- Disable the prefetching LSU, the burst-coalesced LSU, and caching, to infer a pipelined LSU.
- Disable the LSU cache modifier to prevent the compiler from inferring any caching, especially if a cache will not be helpful for your design

However, configuring an LSU solely based on area may compromise throughput. In this tutorial, the access pattern is ideal for the prefetch LSU and it will achieve the same throughput as the burst coalesced, but this is not always the case.

For more details on the descriptions of LSU controls, styles, and modifiers, refer to the LSU Controls section in the [*FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide*](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide).

## Run the `LSU Control` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./lsu_control.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./lsu_control.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./lsu_control.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   lsu_control.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   lsu_control.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   lsu_control.fpga.exe
   ```

### Example Output

```
Kernel throughput with prefetch LSU: 1040.11 MB/s
Kernel throughput with burst-coalesced LSU: 1035.82 MB/s
Kernel throughput without LSU controls: 1040.61 MB/s
PASSED: all kernel results are correct.
```

The throughput observed when running all three kernels, `KernelPrefetch`, `KernelBurst`, and `KernelDefault`, is printed to the standard out. The throughput values are comparable among these; however from the reports, we see an area savings for the `KernelPrefetch`. Therefore, you can use the area savings of the `KernelPrefetch` safely knowing it will not compromise throughput.

> **Note**: The numbers shown are from a compile and run on the Intel® PAC with Intel Arria® 10 GX FPGA. The performance difference will be apparent only when running on FPGA hardware. The emulator and simulator do not reflect the design's hardware memory system performance.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).