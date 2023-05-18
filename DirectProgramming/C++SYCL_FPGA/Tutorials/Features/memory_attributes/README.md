# `Memory Attributes` Sample

This FPGA tutorial demonstrates how to use on-chip memory attributes to control memory structures in your SYCL*-compliant program.

| Area                 | Description
|:--                   |:--
| What you will learn  |  The basic concepts of on-chip memory attributes <br> How to apply memory attributes in your program <br> How to confirm that the memory attributes were respected by the compiler <br> A case study of the type of performance/area trade-offs enabled by memory attributes
| Time to complete     | 30 minutes
| Category             | Concepts and Functionality

## Purpose

For each private or local array in your FPGA device code, the compiler creates a custom memory system in your program's datapath to contain the contents of that array. The compiler has many options to choose from when architecting this on-chip memory structure. Memory attributes are a set of SYCL*-compliant extensions for FPGA that enable you to override the internal compiler heuristics and control kernel memory architecture.

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
It is categorized as a Tier 2 sample that demonstrates a compiler feature.

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

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

## Key Implementation Details

The sample illustrates the following important concepts.

- The basic concepts of on-chip memory attributes.
- How to apply memory attributes in your program.
- How to confirm that the compiler respected the memory attributes.
- A case study of the type of performance/area trade-offs enabled by memory attributes.

### Introduction to Memory Attributes

To maximize kernel throughput, your design's datapath should have stall-free access to all of its memory systems. A memory read or write is said to be *stall-free* if the compiler can prove that it has contention-free access to a memory port. A memory system is stall-free if all of its accesses have this property. Wherever possible, the compiler will try to create a minimum-area, stall-free memory system.

You may prefer a different area performance trade-off for your design, or in some cases the compiler may fail to find the best configuration. In such situations, you can use memory attributes to override the compiler’s decisions and specify the memory configuration you need.

Memory attributes can be applied to any variable or array defined within the kernel and to struct data members in struct declarations. The compiler supports the following memory attributes:

| Memory Attribute                    | Description
|:---                                 |:---
| `intel::fpga_register`              | Forces a variable or array to be carried through the pipeline in registers.
| `intel::fpga_memory("impl_type")`   | Forces a variable or array to be implemented as embedded memory. The optional string parameter `impl_type` can be `BLOCK_RAM` or `MLAB`.
| `intel::numbanks(N)`                | Specifies that the memory implementing the variable or array must have N memory banks.
| `intel::bankwidth(W)`               | Specifies that the memory implementing the variable or array must be W bytes wide.
| `intel::singlepump`                 | Specifies that the memory implementing the variable or array should be clocked at the same rate as the accesses to it.
| `intel::doublepump`                 | Specifies that the memory implementing the variable or array should be clocked at twice the rate as the accesses to it.
| `intel::max_replicates(N)`          | Specifies that a maximum of N replicates should be created to enable simultaneous reads from the datapath.
| `intel::private_copies(N)`          | Specifies that a maximum of N private copies should be created to enable concurrent execution of N pipelined threads.
| `intel::simple_dual_port`           | Specifies that the memory implementing the variable or array should have no port that services both reads and writes.
| `intel::merge("key", "type")`       | Merge two or more variables or arrays in the same scope width-wise or depth-wise. All variables with the same `key` string are merged into the same memory system. The string `type` can be either `width` or `depth`.
| `intel::bank_bits(b0, b1,..., bn)`  | Specifies that the local memory addresses should use bits `(b0, b1,..., bn)` for bank-selection, where `(b0, b1,..., bn)` are indicated in terms of word-addressing. The bits of the local memory address not included in `(b0, b1,..., bn)` will be used for word-selection in each bank.

#### Example 1: Applying Memory Attributes to Private Arrays

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

#### Example 2: Applying Memory Attributes to Struct Data Members

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

### Tutorial Overview

This tutorial demonstrates the trade-offs between choosing a single-pumped and double-pumped memory system for your kernel. We will apply the attributes `[[intel::singlepump]]` and `[[intel::doublepump]]` to the two dimensional array `dict_offset`.

The tutorial enqueues three versions of the same kernel:

- `dict_offset` is single-pumped
- `dict_offset` is double-pumped
- `dict_offset` unconstrained (compiler heuristics choose the memory configuration)

For both single-pumped and double-pumped versions, additional memory attributes direct the compiler to implement `dict_offset` in MLABs (as the size of the array is small), to using `kVec` banks, and to confine the number of replicates in each bank to no more than `kVec`.

### Accesses to `dict_offset`

Array `dict_offset` has the following accesses:

- **Initialization**: It is initialized by copying the contents of global memory `dict_offset_init` using `kVec` writes.
- **Reads** : It is read from `kVec*kVec` times.
- **Writes**: There are `kVec` writes updating the values at some indices.

After all loops are unrolled, the innermost dimension of every access is known at compile time (e.g. `dict_offset[i][k]` becomes `dict_offset[i][0]`, `dict_offset[i][1]`, etc.).

### Banks and Replicates of `dict_offset`

If we partition the memory system so that array elements `dict_offset[:][0]` (where `:` denotes all indices in range) are contained in Bank 0, `dict_offset[:][1]` are contained in Bank 1, and so on, each access is confined to a single bank. This partitioning is achieved by requesting the compiler to generate `kVec` banks.

In total, there are `kVec` reads from each bank. To make these reads stall-free, we request `kVec` replicates per bank so that (if needed) each read can occur simultaneously from a separate replicate. Since all replicates in a bank must contain identical data, a write to a bank must go to all replicates.

For single-pumped memories, each replicate has two physical ports. In the tutorial code, one of these ports is used for writing and one for reading. The compiler must generate `kVec` replicates per bank to create stall-free accesses for `kVec` reads.

For double-pumped memories, each replicate effectively has four ports, three of which are available for reads. Hence, the compiler needs fewer replicates per bank to create stall-free reads. However, this can incur a system f<sub>MAX</sub> penalty.

The choice of attributes will be further discussed in the [Examining the Reports](#examining-the-reports) section.

## Build the `Memory Attributes` Tutorial

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables.
> Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window.
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

## Read the Reports

Locate `report.html` in the `memory_attributes_report.prj/reports/` directory.

Navigate to the Kernel Memory Viewer (System Viewers > Kernel Memory Viewer). In the Kernel Memory List pane, click `dict_offset` under the function `Kernel<N>`, for each of the following:

- **N=0**: unconstrained configuration (compiler's choice)
- **N=1**: single-pumped configuration
- **N=2**: double-pumped configuration

This view provides information about the memory configuration. The user-specified memory attributes are listed in the "Details" pane.

### Comparing the Memory Configurations

For both single-pumped and double-pumped versions of the kernel, the compiler generates `kVec` banks and implements the memory in MLABs, as was requested through memory attributes. The main difference between these two memory systems is the number of replicates within each bank. To see the number of replicates per bank, click any bank label (say Bank 0) under `dict_offset`.

For the single-pumped memory system, the compiler created four replicates per bank, whereas, for the double-pumped memory system, the compiler created two replicates per bank. A single-pumped replicate has two physical ports, and double-pumped replicates have four (effective) physical ports. For this reason, the compiler required twice as many replicates to create a stall-free system in the single-pumped version as compared to the double-pumped version.

### Area Implications

This also means that the FPGA resources needed to generate the stall-free memory systems differ between the two versions. In the report, navigate to the Area Analysis of System view (Area Analysis > Area Analysis of System) and click "Expand All". For the single-pumped version, you can see that the compiler used 32 MLABs to implement the memory system for `dict_offset`, whereas, for the double-pumped version, the compiler used only 16 MLABs. However, the double-pumped version of the memory required additional ALUTs and FFs to implement the double-pumping logic.

In general, double-pumped memories are more area-efficient than single-pumped memories.

### f<sub>MAX</sub> Implications

The use of double-pumped memories can impact the f<sub>MAX</sub> of your system. Double-pumped memories have to be clocked at twice the frequency of the rest of the datapath, and the resulting cross-clock domain transfer can reduce f<sub>MAX</sub>. The effect is particularly pronounced when double-pumping MLABs.

In this tutorial, both the single-pumped and double-pumped versions of the kernel share a single clock domain, so the difference in f<sub>MAX</sub> cannot be directly observed in the report.

If you want to observe the f<sub>MAX</sub> effect, modify the code to enqueue only the single-pumped (or only the double-pumped) version of the kernel. Only the report generated from a full FPGA compile (`make fpga`) will provide f<sub>MAX</sub> information.

The table below summarizes the f<sub>MAX</sub> achieved when compiling single-kernel variants of the tutorial design to an on Intel® PAC with Intel® Arria® 10 GX FPGA.

Variant         | Fmax (MHz) | \# MLABs in `dict_offset`
|:---           |:---        |:---
|Single-pumped  | 307.9      | 32
|Double-pumped  | 200.0      | 16

> **Note**: The numbers reported in the table will vary slightly from compile to compile.

### Trade Offs

There are often many ways to generate a stall-free memory system. As a programmer, the implementation you choose depends on your design constraints.

- If your design is limited by the available memory resources (block RAMs and MLABs), using double-pumped memory systems can help your design fit in the FPGA device.
- If the f<sub>MAX</sub> of your design is limited by double-pumped memory systems in your kernel, forcing all memory systems to be single-pumped might increase the f<sub>MAX</sub>.

## Run the `Memory Attributes` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./memory_attributes.fpga_emu
   ```

2. Run the sample on the FPGA simulator device (the kernel executes on the CPU).
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./memory_attributes.fpga_sim
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP`).
   ```
   ./memory_attributes.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   memory_attributes.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device (the kernel executes on the CPU).
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   memory_attributes.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP`).
   ```
   memory_attributes.fpga.exe
   ```

## Example Output

```
PASSED: all kernel results are correct.
```
You should experiment with the tutorial code. Try to do the following:

- Change the memory implementation type to block RAMs (using `[[intel::fpga_memory("BLOCK_RAM")]]`) or registers (using `[[intel::fpga_register]]`) to see how it affects the area and f<sub>MAX</sub> of the tutorial design.
- Vary `kRows` and/or `kVec` (both in powers of 2) see how it affects the trade-off between single-pumped and double-pumped memories.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).