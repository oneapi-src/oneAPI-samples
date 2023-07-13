# oneAPI Samples for Field Programmable Gate Arrays (FPGAs)

The folders in this area of the oneAPI-sample GitHub repository include tutorials, reference designs, and libraries specific to field programmable gate array (FPGA) features.

You will need the following toolkits and add-ons:

- [Intel® oneAPI Base Toolkit (Base Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html), specifically the Intel® oneAPI DPC++/C++ Compiler.
- [Intel® FPGA Add-On for oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fpga.html).
- Optionally, you might need access to [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/).

>**Note**: The latest versions of code samples on the master branch are not guaranteed to be stable. Use a [stable release version](https://github.com/oneapi-src/oneAPI-samples/tags) of the repository that corresponds to the version of the compiler you are using.

### Understand FPGA Programming

The *Introduction To FPGA Design Concepts* section of the [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) contains information on the basic concepts that are foundational to FPGA programming. Read that section to get the most from these FPGA samples.

## FPGA Repository Structure

This area of the oneAPI-sample repository has a general structure intended to help you find the resources.

- [Tutorials](Tutorials)
  - [GettingStarted](Tutorials/GettingStarted): Contains basic samples to get you through your first compiles.
  - [Features](Tutorials/Features): Contains samples that demonstrate useful compiler features, like loop unrolling.
  - [DesignPatterns](Tutorials/DesignPatterns): Contains samples that show coding patterns to generate efficient hardware usage.
  - [Tools](Tutorials/Tools): Contains sample to demonstrate how to use external debugging tools, like profiling.
- [ReferenceDesigns](ReferenceDesigns): Contains samples that showcase high-performance designs that take advantage of multiple features and design patterns shown in the *Tutorials* section.
- [include](include): Contains commonly used functions wrapped as libraries.

### Sample Categories

To help you understand and use the code samples in a coherent manner, the samples are categorized by the tiers.

- [Tier 1](#tier-1-get-started): Get Started
- [Tier 2](#tier-2-explore-the-fundamentals): Explore the Fundamentals
- [Tier 3](#tier-3-explore-the-advances-techniques): Explore the Advanced Techniques
- [Tier 4](#tier-4-explore-the-reference-designs): Explore the Reference Designs

#### Tier 1: Get Started

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")
   
   tier1 --> tier2 --> tier3 --> tier4
   
   style tier1 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

| Sample                                                    | Category                                             | Description
|:---                                                       |:---                                                  |:---
| [fpga_compile](Tutorials/GettingStarted/fpga_compile)     | [Tutorials/GettingStarted](Tutorials/GettingStarted) | How and why compiling SYCL* code for FPGA differs from CPU or GPU <br> FPGA device image types and when to use them. <br> The compile options used to target FPGA
| [fast_recompile](Tutorials/GettingStarted/fast_recompile) | [Tutorials/GettingStarted](Tutorials/GettingStarted) | Why to separate host and device code compilation in your FPGA project <br> How to use the `-reuse-exe` and device link. <br> Which method to choose for your project
| [fpga_template](Tutorials/GettingStarted/fpga_template) | [Tutorials/GettingStarted](Tutorials/GettingStarted) | Showcases the CMake build system that is used in other code samples, and serves as a template that you can re-use in your own designs.

#### Tier 2: Explore the Fundamentals

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")
   
   tier1 --> tier2 --> tier3 --> tier4
   
   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```


| Sample                                                                                                                        | Category                                             | Description
|:---                                                                                                                           |:---                                                  |:---
| [ac_fixed](Tutorials/Features/ac_fixed)                                                                                       | [Tutorials/Features](Tutorials/Features)             | How different methods of `ac_fixed` number construction affect hardware resource utilization <br> Recommended method for constructing `ac_fixed` numbers in your kernel <br> Accessing and using the `ac_fixed` math library functions <br> Trading off accuracy of results for reduced resource usage on the FPGA
| [ac_int](Tutorials/Features/ac_int)                                                                                           | [Tutorials/Features](Tutorials/Features)             | Using the `ac_int` data type for basic operations <br> Efficiently using the left shift operation <br> Setting and reading certain bits of an `ac_int` number
| [device_global (experimental)](Tutorials/Features/experimental/device_global)                                                 | [Tutorials/Features](Tutorials/Features)             | The basic usage of the `device_global` class <br> How to initialize a `device_global` to non-zero values
| [double_buffering](Tutorials/DesignPatterns/double_buffering)                                                                 | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How and when to implement the double buffering optimization technique
| [explicit_data_movement](Tutorials/DesignPatterns/explicit_data_movement)                                                     | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to explicitly manage the movement of data for the FPGA
| [hostpipes (experimental)](Tutorials/Features/experimental/hostpipes)                                                         | [Tutorials/Features](Tutorials/Features)             | How to use host pipes to send and receive data between a host and the FPGA 
| [kernel_args_restrict](Tutorials/Features/kernel_args_restrict)                                                               | [Tutorials/Features](Tutorials/Features)             | The problem of pointer aliasing and its impact on compiler optimizations. <br> The behavior of the `kernel_args_restrict` attribute and when to use it on your kernel <br> The effect this attribute can have on kernel performance on FPGA
| [loop_coalesce](Tutorials/Features/loop_coalesce)                                                                             | [Tutorials/Features](Tutorials/Features)             | What the `loop_coalesce` attribute does <br> How `loop_coalesce` attribute affects resource usage and loop throughput <br> How to apply the `loop_coalesce` attribute to loops in your program <br> Which loops make good candidates for coalescing
| [loop_fusion](Tutorials/Features/loop_fusion)                                                                                 | [Tutorials/Features](Tutorials/Features)             | Basics of loop fusion <br> The reasons for loop fusion<br/>How to use loop fusion to increase performance <br> Understanding safe application of loop fusion
| [loop_initiation_interval](Tutorials/Features/loop_initiation_interval)                                                       | [Tutorials/Features](Tutorials/Features)             | The f<sub>MAX</sub>-II tradeoff <br> Default behavior of the compiler when scheduling loops <br> How to use `intel::initiation_interval` to attempt to set the II for a loop <br> Scenarios in which `intel::initiation_interval` can be helpful in optimizing kernel performance
| [loop_ivdep](Tutorials/Features/loop_ivdep)                                                                                   | [Tutorials/Features](Tutorials/Features)             | Basics of loop-carried dependencies <br> The notion of a loop-carried dependence distance <br> What constitutes a safe dependence distance <br> How to aid the compiler's dependence analysis to maximize performance
| [loop_unroll](Tutorials/Features/loop_unroll)                                                                                 | [Tutorials/Features](Tutorials/Features)             | Basics of loop unrolling. <br> How to unroll loops in your program <br> Determining the optimal unroll factor for your program
| [max_interleaving](Tutorials/Features/max_interleaving)                                                                       | [Tutorials/Features](Tutorials/Features)             | The basic usage of the `max_interleaving` attribute <br> How the `max_interleaving` attribute affects loop resource use <br> How to apply the `max_interleaving` attribute to loops in your program
| [memory_attributes](Tutorials/Features/memory_attributes)                                                                     | [Tutorials/Features](Tutorials/Features)             | The basic concepts of on-chip memory attributes <br> How to apply memory attributes in your program <br> How to confirm that the memory attributes were respected by the compiler <br> A case study of the type of performance/area trade-offs enabled by memory attributes
| [pipes](Tutorials/Features/pipes)                                                                                             | [Tutorials/Features](Tutorials/Features)             | The basics of using SYCL*-compliant pipes extension for FPGA <br> How to declare and use pipes
| [printf](Tutorials/Features/printf)                                                                                           | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to declare and use `printf` in program
| [register_map_and_streaming_interfaces (experimental)](Tutorials/Features/experimental/register_map_and_streaming_interfaces) | [Tutorials/Features](Tutorials/Features)             | How to specify the kernel invocation interface and kernel argument interfaces
| [pipelined_kernels (experimental)](Tutorials/Features/experimental/pipelined_kernels)                                         | [Tutorials/Features](Tutorials/Features)             | Basics of declaring and launching a pipelined kernel


#### Tier 3: Explore the Advanced Techniques

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

| Sample                                                                            | Category                                             | Description
|:---                                                                               |:---                                                  |:---
| [autorun](Tutorials/DesignPatterns/autorun)                                       | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How and when to use autorun kernels
| [buffered_host_streaming](Tutorials/DesignPatterns/buffered_host_streaming)       | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to optimally stream data between the host and device to maximize throughput
| [compute_units](Tutorials/DesignPatterns/compute_units)                           | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | A design pattern to generate multiple compute units using template metaprogramming
| [dsp_control](Tutorials/Features/dsp_control)                                     | [Tutorials/Features](Tutorials/Features)             | How to apply global DSP control in command-line interface <br> How to apply local DSP control in source code <br> Scope of datatypes and math operations that support DSP control
| [dynamic_profiler](Tutorials/Tools/dynamic_profiler)                              | [Tutorials/Tools](Tutorials/Tools)                   | About the Intel® FPGA dynamic profiler for DPC++ <br> How to set up and use this tool <br> A case study of using this tool to identify performance bottlenecks in pipes
| [fpga_reg](Tutorials/Features/fpga_reg)                                           | [Tutorials/Features](Tutorials/Features)             | How to use the `ext::intel::fpga_reg` extension <br> How `ext::intel::fpga_reg` can be used to re-structure the compiler-generated hardware <br> Situations in which applying  `ext::intel::fpga_reg` might be beneficial
| [io_streaming](Tutorials/DesignPatterns/io_streaming)                             | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to stream data through the FPGA's IO using IO pipes
| [latency_control (experimental)](Tutorials/Features/experimental/latency_control) | [Tutorials/Features](Tutorials/Features)             | How to set latency constraints to pipes and LSUs accesses <br> How to confirm that the compiler respected the latency control directive
| [loop_carried_dependency](Tutorials/DesignPatterns/loop_carried_dependency)       | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | A technique to remove loop carried dependencies from your FPGA device code, and when to apply it
| [lsu_control](Tutorials/Features/lsu_control)                                     | [Tutorials/Features](Tutorials/Features)             | The basic concepts of LSU styles and LSU modifiers <br>  How to use the LSU controls extension to request specific configurations <br>  How to confirm what LSU configurations are implemented <br> A case study of the type of area trade-offs enabled by LSU
| [n_way_buffering](Tutorials/DesignPatterns/n_way_buffering)                       | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How and when to apply the N-way buffering optimization technique
| [onchip_memory_cache](Tutorials/DesignPatterns/onchip_memory_cache)               | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How and when to implement the on-chip memory cache optimization
| [optimization_targets](Tutorials/Features/optimization_targets)                   | [Tutorials/Features](Tutorials/Features)             | How to set optimization targets for your compile</br>How to use the minimum latency optimization target to compile low-latency designs<br>How to manually override underlying controls set by the minimum latency optimization target
| [optimize_inner_loop](Tutorials/DesignPatterns/optimize_inner_loop)               | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to optimize the throughput of an inner loop with a low trip
| [platform_designer](Tutorials/Tools/experimental/platform_designer)               | [Tutorials/Tools](Tutorials/Tools)                   | How to use an IP Component with Intel® Quartus® Prime Pro Edition software suite and Platform Designer
| [pipe_array](Tutorials/DesignPatterns/pipe_array)                                 | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | A design pattern to generate an array of pipes using SYCL* <br> Static loop unrolling through template metaprogramming
| [private_copies](Tutorials/Features/private_copies)                               | [Tutorials/Features](Tutorials/Features)             | The basic usage of the `private_copies` attribute <br> How the `private_copies` attribute affects the throughput and resource use of your FPGA program <br> How to apply the `private_copies` attribute to variables or arrays in your program <br> How to identify the correct `private_copies` factor for your program
| [read_only_cache](Tutorials/Features/read_only_cache)                             | [Tutorials/Features](Tutorials/Features)             | How and when to use the read-only cache feature
| [scheduler_target_fmax](Tutorials/Features/scheduler_target_fmax)                 | [Tutorials/Features](Tutorials/Features)             | The behavior of the `scheduler_target_fmax_mhz` attribute and when to use it <br> The effect this attribute can have on kernel performance on FPGA
| [shannonization](Tutorials/DesignPatterns/shannonization)                         | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to make FPGA-specific optimizations to remove computation from the critical path and improve f<sub>MAX</sub>/II
| [simple_host_streaming](Tutorials/DesignPatterns/simple_host_streaming)           | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to achieve low-latency host-device streaming while maintaining throughput
| [speculated_iterations](Tutorials/Features/speculated_iterations)                 | [Tutorials/Features](Tutorials/Features)             | What the `speculated_iterations` attribute does <br> How to apply the `speculated_iterations` attribute to loops in your program <br> How to determine the optimal number of speculated iterations
| [stall_enable](Tutorials/Features/stall_enable)                                   | [Tutorials/Features](Tutorials/Features)             | What the `use_stall_enable_clusters` attribute does <br> How `use_stall_enable_clusters` attribute affects resource usage and latency <br> How to apply the `use_stall_enable_clusters` attribute to kernels in your program
| [system_profiling](Tutorials/Tools/system_profiling)                              | [Tutorials/Tools](Tutorials/Tools)                   | Summary of profiling tools available for performance optimization <br> About the Intercept Layer for OpenCL™ Applications <br> How to set up and use this tool <br> A case study of using this tool to identify when the double buffering system-level optimization is beneficial
| [triangular_loop](Tutorials/DesignPatterns/triangular_loop)                       | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How and when to apply the triangular loop optimization technique
| [use_library](Tutorials/Tools/use_librar)                                         | [Tutorials/Tools](Tutorials/Tools) | How to integrate Verilog RTL into your oneAPI design directly
| [zero_copy_data_transfer](Tutorials/DesignPatterns/zero_copy_data_transfer)       | [Tutorials/DesignPatterns](Tutorials/DesignPatterns) | How to use SYCL USM host allocations for the FPGA

#### Tier 4: Explore the Reference Designs

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

All the Tier 4 samples are in the [ReferenceDesigns](ReferenceDesigns) category.

| Sample                                                    | Description
|:---                                                       |:---
| [anr](ReferenceDesigns/anr)                               | How to create a parameterizable image processing pipeline to implement an Adaptive Noise Reduction (ANR) algorithm on a FPGA
| [board_test](ReferenceDesigns/board_test)                 | How to test board interfaces to ensure the designed platform provides expected performance
| [cholesky](ReferenceDesigns/cholesky)                     | How to implement high performance matrix Cholesky decomposition on a FPGA
| [cholesky_inversion](ReferenceDesigns/cholesky_inversion) | How to implement high performance Cholesky matrix decomposition on a FPGA
| [crr](ReferenceDesigns/crr)                               | How to implement the Cox-Ross-Rubinstein (CRR) binomial tree model on a FPGA
| [db](ReferenceDesigns/db)                                 | How to accelerate database queries using an FPGA
| [decompress](ReferenceDesigns/decompress)                 | How to implement an efficient GZIP and Snappy decompression engine on a FPGA
| [gzip](ReferenceDesigns/gzip)                             | How to implement a high-performance multi-engine compression algorithm on FPGA
| [matmul](ReferenceDesigns/matmul)                         | How to implement a systolic-array-based high-performance matrix multiplication algorithm on FPGA
| [merge_sort](ReferenceDesigns/merge_sort)                 | How to use the spatial compute of the FPGA to create a merge sort design that takes advantage of thread- and SIMD-level parallelism
| [mvdr_beamforming](ReferenceDesigns/mvdr_beamforming)     | How to create a full, complex system that performs IO streaming using SYCL*-compliant code
| [pca](ReferenceDesigns/pca)                               | How to implement high performance principal component analysis on a FPGA
| [qrd](ReferenceDesigns/qrd)                               | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition algorithm
| [qri](ReferenceDesigns/qri)                               | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition to compute a matrix inversion

#### Start exploring the FPGA code samples with this selection

The following FPGA samples represent a selection of useful tutorials suitable to get you started on your first oneAPI application on the FPGA

| Subject                                   | Sample
|:---                                       |:---
| FPGA Compile Flow                         | [fpga_compile](Tutorials/GettingStarted/fpga_compile)
| Save Development Time                     | [fast_recompile](Tutorials/GettingStarted/fast_recompile)
| Avoid Aliasing of Kernel Arguments        | [kernel_args_restrict](Tutorials/Features/kernel_args_restrict)
| Optimize by Improving Loop Throughput     | [loop_unroll](Tutorials/Features/loop_unroll)
| Transfer Data with Pipes                  | [pipes](Tutorials/Features/pipes)  
| Improve Performance with Double Buffering | [double_buffering](Tutorials/DesignPatterns/double_buffering)

## Build and Run the Samples on Local Development System

Each sample contains a `README.md` file with instructions to build and run the sample. The following sections contain information about configuring your development environment to build and run the samples; in most cases, the sample `README.md` file contains specific instructions.

### Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables.
Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window.
This practice ensures that your compiler, libraries, and tools are ready for development.

>**Note**: For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files

The FPGA samples use many of the headers in the [`DirectProgramming/C++SYCL_FPGA/include`](/DirectProgramming/C++SYCL_FPGA/include) folder.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using instructions for Linux.
 5. (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


### Use Integrated Development Environments (IDEs)

You can compile and run the sample using the Eclipse* IDE (Linux*) and Microsoft Visual Studio* (Windows*). For  on using the IDE integration, see [FPGA Workflows on Third-Party IDEs for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).


### Troubleshooting

If an error occurs when compiling a sample, you can get more details by running `make` with the `VERBOSE=1` argument:
``make VERBOSE=1``

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Performance Disclaimers

Tests document performance of components on a particular test, in specific systems and may not reflect all publicly available security updates. 
Differences in hardware, software, or configuration will affect actual performance. 
Consult other sources of information to evaluate performance as you consider your purchase. 
For complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).
See configuration disclosure for details.
No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software, or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](https://www.intel.com).

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

© Intel Corporation.

## Build and Run the Samples on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

You can specify a FPGA runtime node using a single line script similar to the following example.

```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:fpga_runtime:ppn=2` (lower case L) assigns one full node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes    |Command Options
  |:---               |:---
  |FPGA Compile Time  |`qsub -I -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime       |`qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .`
  |GPU                |`qsub -I -l nodes=1:gpu:ppn=2 -d .`
  |CPU                |`qsub -I -l nodes=1:xeon:ppn=2 -d .`

>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to 24 hours.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the [Intel® oneAPI Base Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/documentation/base-toolkit/).

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured for you, you do not need to set environment variables.

## Documentation

- The [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- The [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.
- The [Intel® oneAPI DPC++/C++ Compiler Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-dpc-c-compiler-release-notes.html).
- The [Migrating OpenCL™ FPGA Designs to SYCL*](https://www.intel.com/content/www/us/en/develop/documentation/migrate-opencl-fpga-designs-to-dpcpp/top.html) guide.
- [Additional FPGA-specific Resources](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/additional-information.html).
- The [Intel® Quartus® Prime Pro and Standard Software User Guides](https://www.intel.com/content/www/us/en/support/programmable/support-resources/design-software/user-guides.html).

