
# LSU Control
This FPGA tutorial demonstrates how to configure the load-store units (LSU) in your DPC++ program using the LSU controls extension.  

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | The basic concepts of LSU styles and LSU modifiers <br>  How to use the LSU controls extension to request specific configurations <br>  How to confirm what LSU configurations are implemented <br> A case study of the type of area trade-offs enabled by LSU
| Time to complete                  | 30 minutes

## Purpose

The Intel® oneAPI DPC++ Compiler creates load-store units (LSU) to access off-chip data. The compiler has many options to choose from when configuring each LSU. The DPC++ LSU controls extension allows you to override the compiler's internal heuristics and control the architecture of each LSU. An introduction to the extension in this tutorial will explain the available options, extension defaults, appropriate use cases, and area trade-offs. 

### LSUs and LSU Styles

An LSU is a block that handles loading and storing data to and from memory. Off-chip memory can have variable latency. To mitigate this, different LSU implementations, referred to as styles, are available.

The two LSU styles used in this tutorial are listed below:
| Burst-Coalesced LSU                                                         |  Prefetching LSU
|---                                                                          |---
| - Dynamically buffers requests until the largest possible burst can be made | - Buffers the next contiguous address of data for future loads
| - Makes efficient accesses to global memory but takes more FPGA resources   | - If the access is not contiguous, the buffer must be flushed
| - Works for both loads and stores                                           | - Works only for loads

The best LSU style depends on the memory access pattern in your design. There are trade-offs for each LSU, so picking a configuration solely based on the area may compromise throughput. In this tutorial, the access pattern is ideal for the prefetching LSU, and it will achieve the same throughput as the burst coalesced LSU, but this is not always the case. 
 
In addition to these two styles, there are also LSU modifiers. LSU modifiers are addons that can be combined with LSU styles, such as caching, which can be combined with the burst-coalesced LSU style.
For more details on LSU modifiers and LSU styles, refer to the Memory Accesses section in the [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide). 

### Introduction to the LSU Control Extension

The class: ```INTEL::lsu``` enables you to control the architecture of the LSU. The class has two member functions, load() and store(), which allow loading from and storing to a global pointer.
The table below outlines the different parameters the LSU control extension provides. These will be respected to the extent possible.
|Control                              | Value                  |Default   |Supports            
---                                   |---                     |---       |---                 
|```INTEL::burst_coalesce<B>      ``` | B is a boolean         | false    |both load & store   
|```INTEL::cache<N>               ``` | N is an integer >=  0  | 0        |only load           
|```INTEL::statically_coalesce<B> ``` | B is a boolean         | true     |both load & store   
|```INTEL::prefetch<B>            ``` | B is a boolean         | false    |only load           
If the default options are used, a pipelined LSU is implemented. 

#### Example: Controlling the prefetch and statically_coalesced parameters

```c++
//Creating typedefs using the LSU controls class 
//for each combination of LSU options desired. 
using PrefetchingLSU = INTEL::lsu<INTEL::prefetch<true>,
                                  INTEL::statically_coalesce<false>>;
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
For more details on the descriptions of LSU controls, styles, and modifiers please refer the to the LSU Controls section in the [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide). 

### Tutorial Code Overview
The compiler selects an LSU configuration based on the design's memory access pattern and dependencies. In some cases, it can be beneficial to explicitly set the configuration using the LSU control extension. If the LSU configurations are invalid, the compiler will emit a warning and choose a valid LSU. This tutorial will highlight a situation where using the extension to specify an LSU is beneficial. 

In the tutorial, there are three kernels with the same body:
|Kernel Name                     | How it loads from the read accessor
---                              |---  
| KernelPrefetch                 | ```INTEL::lsu<INTEL::prefetch<true>>``` 
| KernelBurst                    | ```INTEL::lsu<INTEL::burst_coalesce<true>>``` 
| KernelDefault                  | directly loads data from read accessor, instead of using the ```INTEL::lsu``` class


The kernel design requests data from global memory in a contiguous manner. Therefore, both the prefetching LSU and the burst-coalesced LSU would allow the design to have high throughput. However, the prefetching LSU is highly optimized for such access patterns, especially in situations where we know, at compile time, that such access pattern exists. This will generally lead to significant area savings. As a result, between the two kernels, ```KernelPrefetch``` and ```KernelBurst```, an improvement in area should be observed with ```KernelPrefetch```. The kernel ```KernelDefault``` shows the same design without using the LSU controls extension. This kernel acts as both a baseline and illustrates the difference in syntax between using the LSU controls and not using them.

## Key Concepts
* The basic concepts of LSU styles and LSU configurability
* How to use the LSU controls extension to request specific configurations
* How to confirm what LSU configurations are implemented
* A case study of the type of area trade-offs enabled by the LSU controls extension

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `lsu_control` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

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
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/lsu_control.fpga.tar.gz" download>here</a>.

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
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report: 
     ```
     nmake report
     ``` 
   * An FPGA hardware target is not provided on Windows*. 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.
 
 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `lsu_control.prj/reports/` or `lsu_control_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

### Verifying Selected LSU
To check which LSU is used:

1. Navigate to the Graph Viewer (System Viewers > Graph Viewer). 
   *  In this view, click on any Kernel basic block on the left pane. All blue icons in the graph refer to external memory LSUs. Click on them for more details. 
   *  The word "LD" denotes a load, and the word "ST" denotes the store. Feel free to explore the graphs. 
2. In the Kernel pane, click on the label ```KernelPrefetch.B1``` to see a blue LD icon. 
   *  Hovering over this icon should indicate that a prefetching LSU was inferred. 
3. In the Kernel pane, click on the label ```KernelBurst.B1``` to see a blue LD icon. 
   *  Hovering over this icon should indicate that a burst-coalesced LSU was inferred. 
4. In the Kernel pane, click on the label ```KernelDefault.B1```to see a blue LD icon
   *  Hovering over this icon, which LSU style was inferred?

This view provides additional information about the LSU architecture selected in the details pane.

### Area Analysis
To check area usage:

1. Navigate to Area Analysis of System (Area Analysis > Area Analysis of System). 
   *  In this view, you can see how much of the FPGA resources are being utilized (ALUTs, FFs, RAMs, MLABs, DSPs)
   *  A design that takes less area will use less of these resources   
2. In the center pane, expand the heading of Kernel System
   *  Three kernels should be listed: ```KernelBurst```, ```KernelDefault```, ```KernelPrefetch```
   *  Examine which kernel uses the most resources. 

The ```KernelPrefetch``` should use fewer FPGA resources, as the design is a memory access pattern that lends itself to that LSU style best. 

### Trade-offs

There are many different ways to configure an LSU. As a programmer, the implementation you should choose depends on your design constraints.

If your design is limited by the available FPGA resources, you can try the following:
  - Request a prefetching LSU instead of a burst-coalesced LSU for loads.
  - Disable the prefetching LSU, the burst-coalesced LSU, and caching, to infer a pipelined LSU. 
  - Disable the LSU cache modifier to prevent the compiler from inferring any caching, especially if a cache will not be helpful for your design

However, configuring an LSU solely based on area may compromise throughput. In this tutorial, the access pattern is ideal for the prefetch LSU and it will achieve the same throughput as the burst coalesced, but this is not always the case. 
  
For more details on the descriptions of LSU controls, styles, and modifiers, refer to the LSU Controls section in the [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide). 


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./lsu_control.fpga_emu     (Linux)
     lsu_control.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./lsu_control.fpga         (Linux)
     ```

### Example of Output

```
Kernel throughput with prefetch LSU: 1040.11 MB/s 
Kernel throughput with burst-coalesced LSU: 1035.82 MB/s 
Kernel throughput without LSU controls: 1040.61 MB/s
PASSED: all kernel results are correct.
```

### Discussion of Results

The throughput observed when running all three kernels, ```KernelPrefetch```, ```KernelBurst```, and ```KernelDefault```, is printed to the standard out. The throughput values are comparable among these, however from the reports, we see an area savings for the ```KernelPrefetch```. Therefore, we can leverage the area savings of the ```KernelPrefetch``` knowing we won't compromise throughput. 

Note the numbers are from a compile and run on the Intel® PAC with Intel Arria® 10 GX FPGA.
Note that this performance difference will be apparent only when running on FPGA hardware. The emulator does not reflect the design's hardware performance.
