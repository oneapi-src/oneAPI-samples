
# On-Chip Memory Attributes
This FPGA tutorial demonstrates how to use on-chip memory attributes to control memory structures in your DPC++ program.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               |  The basic concepts of on-chip memory attributes <br> How to apply memory attributes in your program <br> How to confirm that the memory attributes were respected by the compiler <br> A case study of the type of performance/area trade-offs enabled by memory attributes
| Time to complete                  | 30 minutes



## Purpose
For each private or local array in your DPC++ FPGA device code, the Intel® oneAPI DPC++ Compiler creates a custom memory system in your program's datapath to contain the contents of that array. The compiler has many options to choose from when architecting this on-chip memory structure. Memory attributes are a set of DPC++ extensions for FPGA that enable you to override the compiler's internal heuristics and control the kernel's memory architecture.

### Introduction to Memory Attributes

To maximize kernel throughput, your design's datapath should have stall-free access to all of its memory systems. A memory read or write is said to be *stall-free* if the compiler can prove that it has contention-free access to a memory port. A memory system is stall-free if all of its accesses have this property. Wherever possible, the compiler will try to create a minimum-area, stall-free memory system.

You may prefer a different area performance trade-off for your design, or in some cases the compiler may fails to find the best configuration. In such situations, you can use memory attributes to override the compiler’s decisions and specify the memory configuration you need.

Memory attributes can be applied to any variable or array defined within the kernel and to struct data members in struct declarations. The compiler supports the following memory attributes:

| Memory Attribute                 | Description
---                                |---
| `intel::fpga_register`           | Forces a variable or array to be carried through the pipeline in registers.
| `intel::fpga_memory("impl_type")`| Forces a variable or array to be implemented as embedded memory. The optional string parameter `impl_type` can be `BLOCK_RAM` or `MLAB`.
| `intel::numbanks(N)`             | Specifies that the memory implementing the variable or array must have N memory banks.
| `intel::bankwidth(W)`            | Specifies that the memory implementing the variable or array must be W bytes wide.
| `intel::singlepump`              | Specifies that the memory implementing the variable or array should be clocked at the same rate as the accesses to it.
| `intel::doublepump`              | Specifies that the memory implementing the variable or array should be clocked at twice the rate as the accesses to it.
| `intel::max_replicates(N)`       | Specifies that a maximum of N replicates should be created to enable simultaneous reads from the datapath.
| `intel::private_copies(N)`       | Specifies that a maximum of N private copies should be created to enable concurrent execution of N pipelined threads.
| `intel::simple_dual_port`        | Specifies that the memory implementing the variable or array should have no port that services both reads and writes.
| `intel::merge("key", "type")`    | Merge two or more variables or arrays in the same scope width-wise or depth-wise. All variables with the same `key` string are merged into the same memory system. The string `type` can be either `width` or `depth`.
| `intel::bank_bits(b0, b1,..., bn)`  | Specifies that the local memory addresses should use bits `(b0, b1,..., bn)` for bank-selection, where `(b0, b1,..., bn)` are indicated in terms of word-addressing. The bits of the local memory address not included in `(b0, b1,..., bn)` will be used for word-selection in each bank.


#### Example 1: Applying memory attributes to private arrays
```c++
q.submit([&](handler &h) {
  h.single_task<class Example1>([=]() {
    // Create a kernel memory 8 bytes wide (2 integers per memory word)
    // and split the contents into 2 banks (each bank will contain 32
    // integers in 16 memory words).
    [[intel::bankwidth(8), intel::numbanks(2)]] int a[64];

    // Force array 'b' to be carried live in the data path using
    // registers.
    [[intel::fpga_register]] int b[64];

    // Merge 'mem_A' and 'mem_B' width-wise so that they are mapped
    // to the same kernel memory system,
    [[intel::merge("mem", "width")]] unsigned short mem_A[64];
    [[intel::merge("mem", "width")]] unsigned short mem_B[64];

    // ...
  });
});

```

#### Example 2: Applying memory attributes to struct data members
```c++
// Memory attributes can be specified for struct data members
// within the struct declaration.
struct State {
  [[intel::numbanks(2)]]      int mem[64];
  [[intel::fpga_register]]    int reg[8];
};

q.submit([&](handler &h) {
  h.single_task<class Example2>([=]() {
    // The compiler will create two memory systems from S1:
    //  - S1.mem[64] implemented in kernel memory that has 2 banks
    //  - S1.reg[8] implemented in registers
    State S1;

    // In this case, we have attributes on struct declaration as
    // well as struct instantiation. When this happens, the outer
    // level attribute takes precedence. Here, the compiler will
    // generate a single memory system for S2, which will have 4
    // banks.
    [[intel::numbanks(4)]] State S2;

    // ...
  });
});

```

### Tutorial Code Overview
This tutorial demonstrates the trade-offs between choosing a single-pumped and double-pumped memory system for your kernel. We will apply the attributes `[[intel::singlepump]]` and `[[intel::doublepump]]` to the two dimensional array `dict_offset`.

The tutorial enqueues three versions of the same kernel:
* `dict_offset` is single-pumped
* `dict_offset` is double-pumped
* `dict_offset` unconstrained (compiler heuristics choose the memory configuration)

For both single-pumped and double-pumped versions, additional memory attributes direct the compiler to implement `dict_offset` in MLABs (as the size of the array is small), to using `kVec` banks, and to confine the number of replicates in each bank to no more than `kVec`.

### Accesses to `dict_offset`

Array `dict_offset` has the following accesses:

 * **Initialization**: It is initialized by copying the contents of global memory `dict_offset_init` using `kVec` writes.
 * **Reads** : It is read from `kVec*kVec` times.
 * **Writes**: There are `kVec` writes updating the values at some indices.

After all loops are unrolled, the innermost dimension of every access is known at compile time (e.g. `dict_offset[i][k]` becomes `dict_offset[i][0]`, `dict_offset[i][1]`, etc.).

### Banks and replicates of `dict_offset`

If we partition the memory system such that array elements `dict_offset[:][0]` (where `:` denotes all indices in range) are contained in Bank 0, `dict_offset[:][1]` are contained in Bank 1, and so on, each access is confined to a single bank. This partitioning is achieved by requesting the compiler to generate `kVec` banks.

In total, there are `kVec` reads from each bank. To make these reads stall-free, we request `kVec` replicates per bank so that (if needed) each read can occur simultaneously from a separate replicate. Since all replicates in a bank must contain identical data, a write to a bank must go to all replicates.

For single-pumped memories, each replicate has two physical ports. In the tutorial code, one of these ports is used for writing and one for reading. The compiler must generate `kVec` replicates per bank to create stall-free accesses for `kVec` reads.

For double-pumped memories, each replicate effectively has four ports, three of which are available for reads. Hence, the compiler needs fewer replicates per bank to create stall-free reads. However, this can incur a system f<sub>MAX</sub> penalty.

The choice of attributes will be further discussed in the [Examining the Reports](#examining-the-reports) section.


## Key Concepts
* The basic concepts of on-chip memory attributes
* How to apply memory attributes in your program
* How to confirm that the compiler respected the memory attributes
* A case study of the type of performance/area trade-offs enabled by memory attributes

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `memory_attributes` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

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
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/memory_attributes.fpga.tar.gz" download>here</a>.

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
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=<board-support-package>:<board-variant>
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
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `memory_attributes_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the Kernel Memory Viewer (System Viewers > Kernel Memory Viewer). In the Kernel Memory List pane, click on `dict_offset` under the function `Kernel<N>`, for each of
* N=0 : unconstrained configuration (compiler's choice)
* N=1 : single-pumped configuration
* N=2 : double-pumped configuration

This view provides information about the memory configuration. The user-specified memory attributes are listed in the "Details" pane.

### Comparing the memory configurations

For both single-pumped and double-pumped versions of the kernel, the compiler generates `kVec` banks and implements the memory in MLABs, as was requested through memory attributes. The main difference between these two memory systems is the number of replicates within each bank. To see the number of replicates per bank, click any bank label (say Bank 0) under `dict_offset`.

For the single-pumped memory system, the compiler created four replicates per bank, whereas, for the double-pumped memory system, the compiler created two replicates per bank. A single-pumped replicate has two physical ports, and double-pumped replicates have four (effective) physical ports. For this reason, the compiler required twice as many replicates to create a stall-free system in the single-pumped version as compared to the double-pumped version.

### Area implications

This also means that the FPGA resources needed to generate the stall-free memory systems differ between the two versions. In the report, navigate to the Area Analysis of System view (Area Analysis > Area Analysis of System) and click "Expand All". For the single-pumped version, you can see that the compiler used 32 MLABs to implement the memory system for `dict_offset`, whereas, for the double-pumped version, the compiler used only 16 MLABs. However, the double-pumped version of the memory required additional ALUTs and FFs to implement the double-pumping logic.

In general, double-pumped memories are more area-efficient than single-pumped memories.

### f<sub>MAX</sub> implications

The use of double-pumped memories can impact the f<sub>MAX</sub> of your system. Double-pumped memories have to be clocked at twice the frequency of the rest of the datapath, and the resulting cross-clock domain transfer can reduce f<sub>MAX</sub>. The effect is particularly pronounced when double-pumping MLABs.

In this tutorial, both the single-pumped and double-pumped versions of the kernel share a single clock domain, so the difference in f<sub>MAX</sub> cannot be directly observed in the report.

If you want to observe the f<sub>MAX</sub> effect, modify the code to enqueue only the single-pumped (or only the double-pumped) version of the kernel. Only the report generated from a full FPGA compile (`make fpga`) will provide f<sub>MAX</sub> information.

The table below summarizes the f<sub>MAX</sub> achieved when compiling single-kernel variants of the tutorial design to an on Intel® PAC with Intel® Arria® 10 GX FPGA.

Variant  | Fmax (MHz) | \# MLABs in `dict_offset`
------------- | ------------- | --------
Single-pumped  | 307.9 | 32
Double-pumped  | 200.0 | 16

Note that the numbers reported in the table will vary slightly from compile to compile.

### Trade-offs
There are often many ways to generate a stall-free memory system. As a programmer, the implementation you choose depends on your design constraints.

 - If your design is limited by the available memory resources (block RAMs and MLABs), using double-pumped memory systems can help your design fit in the FPGA device.
 - If the f<sub>MAX</sub> of your design is limited by double-pumped memory systems in your kernel, forcing all memory systems to be single-pumped might increase the f<sub>MAX</sub>.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./memory_attributes.fpga_emu     (Linux)
     memory_attributes.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./memory_attributes.fpga         (Linux)
     ```

### Example of Output
```
PASSED: all kernel results are correct.
```

### Discussion

Feel free to experiment further with the tutorial code. You can:
 - Change the memory implementation type to block RAMs (using `[[intel::fpga_memory("BLOCK_RAM")]]`) or registers (using `[[intel::fpga_register]]`) to see how it affects the area and f<sub>MAX</sub> of the tutorial design.
 - Vary `kRows` and/or `kVec` (both in powers of 2) see how it affects the trade-off between single-pumped and double-pumped memories.

