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

- [Tier 1](#tier-1:get-started): Get Started
- [Tier 2](#tier-2:explore-the-fundametals): Explore the Fundamentals
- [Tier 3](#tier-3:explore-the-advanced-techniques): Explore the Advanced Techniques
- [Tier 4](#tier-4:explore-the-reference-designs): Explore the Reference Designs

#### Tier 1: Get Started

| Sample                                                    | Category       | Description
|:---                                                       |:---            |:---
| [fpga_compile](Tutorials/GettingStarted/fpga_compile)     | Get Started    | How and why compiling SYCL* code for FPGA differs from CPU or GPU <br> FPGA device image types and when to use them. <br> The compile options used to target FPGA
| [fast_recompile](Tutorials/GettingStarted/fast_recompile) | Get Started    | Why to separate host and device code compilation in your FPGA project <br> How to use the `-reuse-exe` and device link. <br> Which method to choose for your project

#### Tier 2: Explore the Fundamentals

| Sample                                                                    | Category       | Description
|:---                                                                       |:---            |:---
| [double_buffering](Tutorials/DesignPatterns/double_buffering)             | Design Pattern | How and when to implement the double buffering optimization technique
| [explicit_data_movement](Tutorials/DesignPatterns/explicit_data_movement) | Design Pattern | How to explicitly manage the movement of data for the FPGA
| [kernel_args_restrict](Tutorials/Features/kernel_args_restrict)           | Basic Feature  | The problem of pointer aliasing and its impact on compiler optimizations. <br> The behavior of the `kernel_args_restrict` attribute and when to use it on your kernel <br> The effect this attribute can have on kernel performance on FPGA
| [loop_ivdep](Tutorials/Features/loop_ivdep)                               | Basic Feature  | Basics of loop-carried dependencies <br> The notion of a loop-carried dependence distance <br> What constitutes a safe dependence distance <br> How to aid the compiler's dependence analysis to maximize performance <br> **Warning**: This looks like a Tier 3 sample.
| [loop_unroll](Tutorials/Features/loop_unroll)                             | Basic Feature  | Basics of loop unrolling. <br> How to unroll loops in your program <br> Determining the optimal unroll factor for your program
| [pipes](Tutorials/Features/pipes)                                         | Basic Feature  | The basics of using SYCL*-compliant pipes extension for FPGA <br> How to declare and use pipes
| [printf](Tutorials/Features/printf)                                       | Design Pattern | How to declare and use `printf` in program

#### Tier 3: Explore the Advances Techniques

| Sample                                                                            | Category          | Description
|:---                                                                               |:---               |:---
| [ac_fixed](Tutorials/Features/ac_fixed)                                           | Basic Feature     | How different methods of `ac_fixed` number construction affect hardware resource utilization <br> Recommended method for constructing `ac_fixed` numbers in your kernel <br> Accessing and using the `ac_fixed` math library functions <br> Trading off accuracy of results for reduced resource usage on the FPGA
| [ac_int](Tutorials/Features/ac_int)                                               | Basic Feature     | Using the `ac_int` data type for basic operations <br> Efficiently using the left shift operation <br> Setting and reading certain bits of an `ac_int` number
| [autorun](Tutorials/DesignPatterns/autorun)                                       | Design Pattern    | How and when to use autorun kernels
| [buffered_host_streaming](Tutorials/DesignPatterns/buffered_host_streaming)       | Design Pattern    | How to optimally stream data between the host and device to maximize throughput
| [compute_units](Tutorials/DesignPatterns/compute_units)                           | Design Pattern    | A design pattern to generate multiple compute units using template metaprogramming
| [dsp_control](Tutorials/Features/dsp_control)                                     | Advanced Feature  | How to apply global DSP control in command-line interface <br> How to apply local DSP control in source code <br> Scope of datatypes and math operations that support DSP control
| [dynamic_profiler](Tutorials/Tools/dynamic_profiler)                              | Tools             | About the Intel® FPGA dynamic profiler for DPC++ <br> How to set up and use this tool <br> A case study of using this tool to identify performance bottlenecks in pipes
| [fpga_reg](Tutorials/Features/fpga_reg)                                           | Advanced Feature  | How to use the `ext::intel::fpga_reg` extension <br> How `ext::intel::fpga_reg` can be used to re-structure the compiler-generated hardware <br> Situations in which applying  `ext::intel::fpga_reg` might be beneficial
| [io_streaming](Tutorials/DesignPatterns/io_streaming)                             | Design Pattern    | How to stream data through the FPGA's IO using IO pipes
| [latency_control (experimental)](Tutorials/Features/experimental/latency_control) | Advanced Feature  | How to set latency constraints to pipes and LSUs accesses <br> How to confirm that the compiler respected the latency control directive
| [loop_carried_dependency](Tutorials/DesignPatterns/loop_carried_dependency)       | Design Pattern    | A technique to remove loop carried dependencies from your FPGA device code, and when to apply it
| [loop_coalesce](Tutorials/Features/loop_coalesce)                                 | Basic Feature     | What the `loop_coalesce` attribute does <br> How `loop_coalesce` attribute affects resource usage and loop throughput <br> How to apply the `loop_coalesce` attribute to loops in your program <br> Which loops make good candidates for coalescing
| [loop_fusion](Tutorials/Features/loop_fusion)                                     | Basic Feature     | Basics of loop fusion <br> The reasons for loop fusion<br/>How to use loop fusion to increase performance <br> Understanding safe application of loop fusion
| [loop_initiation_interval](Tutorials/Features/loop_initiation_interval)           | Basic Feature     | The f<sub>MAX</sub>-II tradeoff <br> Default behavior of the compiler when scheduling loops <br> How to use `intel::initiation_interval` to attempt to set the II for a loop <br> Scenarios in which `intel::initiation_interval` can be helpful in optimizing kernel performance
| [lsu_control](Tutorials/Features/lsu_control)                                     | Advanced Feature  | The basic concepts of LSU styles and LSU modifiers <br>  How to use the LSU controls extension to request specific configurations <br>  How to confirm what LSU configurations are implemented <br> A case study of the type of area trade-offs enabled by LSU
| [max_interleaving](Tutorials/Features/max_interleaving)                           | Basic Feature     | The basic usage of the `max_interleaving` attribute <br> How the `max_interleaving` attribute affects loop resource use <br> How to apply the `max_interleaving` attribute to loops in your program
| [mem_channel](Tutorials/Features/mem_channel)                                     | Advanced Feature  | How and when to use the `mem_channel` buffer property and the `-Xsno-interleaving` flag
| [memory_attributes](Tutorials/Features/memory_attributes)                         | Basic Feature     | The basic concepts of on-chip memory attributes <br> How to apply memory attributes in your program <br> How to confirm that the memory attributes were respected by the compiler <br> A case study of the type of performance/area trade-offs enabled by memory attributes
| [n_way_buffering](Tutorials/DesignPatterns/n_way_buffering)                       | Design Pattern    | How and when to apply the N-way buffering optimization technique
| [onchip_memory_cache](Tutorials/DesignPatterns/onchip_memory_cache)               | Design Pattern    | How and when to implement the on-chip memory cache optimization
| [optimize_inner_loop](Tutorials/DesignPatterns/optimize_inner_loop)               | Design Pattern    | How to optimize the throughput of an inner loop with a low trip
| [pipe_array](Tutorials/DesignPatterns/pipe_array)                                 | Design Pattern    | A design pattern to generate an array of pipes using SYCL* <br> Static loop unrolling through template metaprogramming
| [private_copies](Tutorials/Features/private_copies)                               | Advanced Feature  | The basic usage of the `private_copies` attribute <br> How the `private_copies` attribute affects the throughput and resource use of your FPGA program <br> How to apply the `private_copies` attribute to variables or arrays in your program <br> How to identify the correct `private_copies` factor for your program
| [read_only_cache](Tutorials/Features/read_only_cache)                             | Advanced Feature  | How and when to use the read-only cache feature
| [scheduler_target_fmax](Tutorials/Features/scheduler_target_fmax)                 | Advanced Feature  | The behavior of the `scheduler_target_fmax_mhz` attribute and when to use it <br> The effect this attribute can have on kernel performance on FPGA
| [shannonization](Tutorials/DesignPatterns/shannonization)                         | Design Pattern    | How to make FPGA-specific optimizations to remove computation from the critical path and improve f<sub>MAX</sub>/II
| [simple_host_streaming](Tutorials/DesignPatterns/simple_host_streaming)           | Design Pattern    | How to achieve low-latency host-device streaming while maintaining throughput
| [speculated_iterations](Tutorials/Features/speculated_iterations)                 | Advanced Feature  | What the `speculated_iterations` attribute does <br> How to apply the `speculated_iterations` attribute to loops in your program <br> How to determine the optimal number of speculated iterations
| [stall_enable](Tutorials/Features/stall_enable)                                   | Advanced Feature  | What the `use_stall_enable_clusters` attribute does <br> How `use_stall_enable_clusters` attribute affects resource usage and latency <br> How to apply the `use_stall_enable_clusters` attribute to kernels in your program
| [system_profiling](Tutorials/Tools/system_profiling)                              | Tools             | Summary of profiling tools available for performance optimization <br> About the Intercept Layer for OpenCL™ Applications <br> How to set up and use this tool <br> A case study of using this tool to identify when the double buffering system-level optimization is beneficial
| [triangular_loop](Tutorials/DesignPatterns/triangular_loop)                       | Design Pattern    | How and when to apply the triangular loop optimization technique
| [zero_copy_data_transfer](Tutorials/DesignPatterns/zero_copy_data_transfer)       | Design Pattern    | How to use SYCL USM host allocations for the FPGA

#### Tier 4: Explore the Reference Designs

All the Tier 4 samples are in the Reference Design category.

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
| [merge_sort](ReferenceDesigns/merge_sort)                 | How to use the spatial compute of the FPGA to create a merge sort design that takes advantage of thread- and SIMD-level parallelism
| [mvdr_beamforming](ReferenceDesigns/mvdr_beamforming)     | How to create a full, complex system that performs IO streaming using SYCL*-compliant code
| [qrd](ReferenceDesigns/qrd)                               | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition algorithm
| [qri](ReferenceDesigns/qri)                               | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition to compute a matrix inversion

## Build and Run the Samples on Local Development System

Each sample contains a `README.md` file with instructions to build and run the sample. The following sections contain information about configuring your development environment to build and run the samples; in most cases, the sample `README.md` file contains specific instructions.

### Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

>**Note**: For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files

All the FPGA code samples include the `dpc_common.hpp` header. The include folder is at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system. You might need to use some of the resources from this location to build the sample.

>**Note**: You can get the common resources from the [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples/tree/master/common) GitHub repository.

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

You can compile and run the sample using the Eclipse* IDE (Linux*) and Microsoft Visual Studio* (Windows*). For  on using the IDE integration, see [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).


### Troubleshooting

If an error occurs when compiling a sample, you can get more details by running `make` with the `VERBOSE=1` argument:
``make VERBOSE=1``

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.



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
  |GPU	              |`qsub -I -l nodes=1:gpu:ppn=2 -d .`
  |CPU	              |`qsub -I -l nodes=1:xeon:ppn=2 -d .`

>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to 12 hours.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the [Intel® oneAPI Base Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/documentation/base-toolkit/).

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured for you, you do not need to set environment variables.

## Documentation:

- The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA.
- The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- The [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.
- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL by suggesting a series of samples.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).