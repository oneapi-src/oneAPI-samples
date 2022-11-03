# `Merge Sort` Sample

This reference design demonstrates a merge sort algorithm on an FPGA.

| Optimized for         | Description
|:---                   |:---
| What you will learn   | How to use the spatial compute of the FPGA to create a merge sort design that takes advantage of thread- and SIMD-level parallelism.
| Time to complete      | 1 hour
| Category              | Reference Designs and End to End


## Purpose

This FPGA reference design demonstrates a merge sort design with multiple parameters that utilizes the spatial computing of the FPGA.

>**Note**: See the [merge sort](https://en.wikipedia.org/wiki/Merge_sort) Wikipedia article for more information.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler <br> Intel® FPGA Add-on for oneAPI Base Toolkit

## Key Implementation Details

### Merge Sort Details

This section describes how the merge sort design is structured and how it takes advantage of the spatial compute of the FPGA.

The figure below shows the conceptual view of the merge sort design to the user. The user streams data into a SYCL pipe (`InPipe`) and, after some delay, the elements are streamed out of a SYCL pipe (`OutPipe`), in sorted order. The number of elements that the merge sort design is capable of sorting is a runtime parameter, but it must be a power of 2. However, this restriction can be worked around by padding the input stream with min/max elements, depending on the direction of the sort (smallest-to-largest vs largest-to-smallest). This technique is demonstrated in this design (see the `fpga_sort` function in *main.cpp*).

![sort_api](sort_api.png)

The basis of the merge sort design is what we call a *merge unit*, which is shown in the figure below. A single merge unit streams in two sorted lists of size `count` in parallel and merges them into a single sorted list of size `2*count`. The lists are streamed in from device memory (e.g., DDR or HBM) by two `Produce` kernels. The `Consume` kernel can stream data out to either a SYCL pipe or to device memory.

![merge_unit](merge_unit.png)

A single merge unit requires `lg(N)` iterations to sort `N` elements. This requires the host to enqueue `lg(N)` iterations of the merge unit kernels that merge sublists of size {`1`, `2`, `4`, ...} into larger lists of size {`2`, `4`, `8`, ...}, respectively. This results in a timeline that looks like the figure below.

![basic_runtime_graph](basic_runtime_graph.png)

To achieve SIMD-level (**S**ingle **I**nstruction **M**ultiple **D**ata) parallelism, we enhance the merge unit to merge `k` elements per cycle. The figure below illustrates how this is done. In the following discussion, we will assume that we are sorting from smallest-to-largest, but the logic is very similar for sorting largest-to-smallest and is easily configurable at compile time in this design.

The merge unit looks at the two inputs of size `k` coming from the `ProduceA` and `ProduceB` kernels (in the figure below, `k=4`) and compares the first elements of each set; remember, the set of `k` elements are already sorted, so we are comparing the smallest elements of the set. Whichever set of elements has the *smaller of the smallest elements* is chosen and combined with `k` other elements from the `feedback` path. These `2*k` elements go through a merge sort network that sorts them in a single cycle. After the `2*k` elements are sorted, the smallest `k` elements are sent to the output (to the `Consume` kernel) and the largest `k` elements are fed back into the sorting network (the `feedback` path in the figure below), and the process repeats. This allows the merge unit to process `k` elements per cycle in the steady state. Note that `k` must be a power of 2. 

>**Note**: You can find more information about this design in the paper *[A High Performance FPGA-Based Sorting Accelerator with a Data Compression Mechanism](https://www.researchgate.net/publication/316604001_A_High_Performance_FPGA-Based_Sorting_Accelerator_with_a_Data_Compression_Mechanism)* by Ryohei Kobayashi and Kenji Kise.

![way_merge_unit](k-way_merge_unit.png)

To achieve thread-level parallelism, the merge sort design accepts a template parameter, `units`, which allows one to instantiate multiple instances of the merge unit, as shown in the figure below. Before the merge units start processing data, the incoming data coming from the input pipe is sent through a bitonic sorting network and written to the temporary buffer partitions in device memory. This sorting network sorts `k` elements per cycle in the steady state. Choosing the number of merge units is an area-performance tradeoff (note: the number of instantiated merge units must be a power of 2). Each merge unit sorts an `N/units`-sized partition of the input data in parallel.

![parallel_tree_bitonic_k-way](parallel_tree_bitonic_k-way.png)

After the merge units sort their `N/units`-sized partition, the partitions of each unit must be reduced into a single sorted list. There are two options to do this: (1) reuse the merge units to perform `lg(units)` more iterations to sort the partitions, or (2) create a merge tree to reduce the partitions into a single sorted list. Option (1) saves area at the expense of performance, since it has to perform additional sorting iterations. Option (2), which we choose for this design, improves performance by creating a merge tree to reduce the final partitions into a single sorted list. The `Merge` kernels in the merge tree (shown in the figure above) use the same kernel code that is used in the `Merge` kernel of the merge unit, which means they too can merge `k` elements per cycle. Once the merge units perform their last iteration, they output to a pipe (instead of writing to device memory) that feeds the merge tree.

### Source Code

The following source files can be found in the `src/` sub-directory.

| File                   | Description
|:---                    |:---
|`main.cpp`              | Contains the `main()` function and the top-level interfaces.
|`merge_sort.hpp`        | The function to submit all of the merge sort kernels (`SortingNetwork`, `Produce`, `Merge`, and `Consume`).
|`consume.hpp`           | The `Consume` kernel for the merge unit. This kernel reads from an input pipe and writes out to either a different output pipe, or to device memory.
|`merge.hpp`             | The `Merge` kernel for the merge unit and the merge tree. This kernel streams in two sorted lists, merges them into a single sorted list of double the size, and streams the data out a pipe.
|`produce.hpp`           | The `Produce` kernel for the merge unit. This kernel reads from input pipes or performs strided reads from device memory and writes the data to an output pipe.
|`sorting_networks.hpp`  | Contains all of the code relevant to sorting networks, including the `SortingNetwork` kernel, as well as the `BitonicSortingNetwork` and `MergeSortNetwork` helper functions.

For `constexpr_math.hpp`, `pipe_utils.hpp`, and `unrolled_loop.hpp` see the README in the `../include/` directory of the FPGA section of the repository.

### Additional Documentation

- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Merge Sort` Design

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
       ```
       make fpga_emu
       ```
   2. Generate the HTML performance report.
       ```
       make report
       ```
      The report resides at `merge_sort_report.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
       ```
       make fpga
       ```

   (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/merge_sort.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/merge_sort.fpga.tar.gz).

### On Windows*

> **Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the HTML performance report.
      ```
      nmake report
      ```
      The report resides at `merge_sort_report.a.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Merge Sort` Program

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./merge_sort.fpga_emu
   ```
2. Run the sample on the FPGA device.
   ```
   ./merge_sort.fpga
   ```
### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   merge_sort.fpga_emu.exe
   ```
2. Run the sample on the FPGA device.
   ```
   merge_sort.fpga.exe
   ```
### Build and Run the Sample on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

Use the Linux instructions to build and run the program.

You can specify an FPGA runtime node using a single line script similar to the following example.

```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:fpga_runtime:ppn=2` (lower case L) assigns one full node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes           |Command Options
  |:---                      |:---
  |FPGA Compile Time         |`qsub -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime (Arria 10)   |`qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d .`
  |FPGA Runtime (Stratix 10) |`qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d .`
  |GPU	                    |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	                    |`qsub -l nodes=1:xeon:ppn=2 -d .`

>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **24 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.


## Example Output

>**Note**: When running on the FPGA emulator, the *Execution time* and *Throughput* values do not reflect the design's actual hardware performance.

```
Running sort 17 times for an input size of 16777216 using 8 4-way merge units
Streaming data from device memory
Execution time: 24.7522 ms
Throughput: 646.408 Melements/s
PASSED
```
>**Note**: The performance numbers above were achieved using the Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX); your results may vary.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).