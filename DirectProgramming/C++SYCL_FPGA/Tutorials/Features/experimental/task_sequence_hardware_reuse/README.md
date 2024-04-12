# `Hardware Reuse` Sample

This sample is a tutorial that demonstrates how to reuse hardware in your FPGA designs by using loops and task sequences.

| Area                 | Description
|:--                   |:--
| What you will learn  | How to code your FPGA SYCL program to reuse hardware that has same functionality
| Time to complete     | 30 minutes
| Category             | Concepts and Functionality

## Purpose

In normal operation, the oneAPI FPGA compiler automatically in-lines repeated function calls, creating multiple instances of hardware that can run in parallel. This in-lining strategy is necessary to achieve high throughput, but in cases where you want to minimize area utilization, you may wish to sacrifice throughput and allow certain functions to be re-used. This sample demonstrates two strategies to re-use hardware in your designs:

1. Calling a function in a loop will cause it to only be instantiated once
2. Calling a function in a task sequence lets you re-use a function in cases where it is not practical to re-use it in a loop.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> To use the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.
>
> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

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

This sample illustrates some key concepts:
- Simple hardware reuse by invoking the function task in a 'for' loop.
- More flexible hardware reuse by invoking the function task using a task sequence wrapper object.

This tutorial demonstrates a design that computes the 'square root of a dot product' multiple times. 

<table>
<tr>
<td>

```c++
D3Vector new_item, item;

item = InputPipeA::read();
new_item[0] = OpSqrt(item, coef1);
item = InputPipeA::read();
new_item[1] = OpSqrt(item, coef1);
item = InputPipeA::read();
new_item[2] = OpSqrt(item, coef1);

OutputPipeZ::write(OpSqrt(new_item, coef2));
```

</td>

<td>
<img src="assets/hardware_reuse_naive.svg" />
</td>

</tr>
</table>

Without resource sharing, multiple identical compute blocks for square root of dot product are generated on the FPGA. However by sharing the compute block using a loop or task sequence, we can tell the compiler to re-use the compute block, and avoid replicating the compute block hardware on the FPGA. The important trade-off made by resource sharing is that multiple invocations of the same task block each other, so the II of the top level component will increase.

### Resource sharing with a loop
Instead of invoking the square root function one by one, you may re-use the function by calling it in a loop. 

<table>
<tr>
<td>

```c++
struct VectorOp{
  int len;

  void operator()() const{
    constexpr D3Vector coef1 = {0.2, 0.3, 0.4};
    constexpr D3Vector coef2 = {0.6, 0.7, 0.8};

    D3Vector new_item;

    // Calling OpSqrt() in a loop will re-use it
    for (int i = 0; i < len; i++){
      D3Vector item = InputPipeA::read();
      new_item[i] = OpSqrt(item, coef1);
    }

    // Another square root block 
    // will be generated for this function call
    OutputPipeZ::write(OpSqrt(new_item, coef2));
  }
};
```

</td>

<td>
<img src="assets/hardware_reuse_loop.svg" />
</td>

</tr>
</table>

### Resource sharing with a task sequence object
Each task sequence class object represents a specific instantiation of FPGA hardware to perform the task function operation.
Launching tasks via the `async()` function calls on the same object results in the reuse of that object's hardware. Thus, you can control the reuse or replication of FPGA hardware by the number of `task_sequence` objects you declare. Since object lifetime is confined to the scope in which the `task_sequence` object is created, carefully declare your object in the scope in which you intend to perform its reuse.

In the `task_sequence` directory, the device code declares the task sequence object once, and invokes it inside the loop, and at the return point.

<table>
<tr>
<td>

```c++
struct VectorOp{
  int len;

  void operator()() const{
    constexpr D3Vector coef1 = {0.2, 0.3, 0.4};
    constexpr D3Vector coef2 = {0.6, 0.7, 0.8};

    D3Vector new_item;

    // Object declarations of a parameterized task_sequence 
    // class must be local, which means global declarations
    // and dynamic allocations are not allowed.
    // Declare the task sequence object outside the for loop 
    // so that the hardware can be shared at the return point.
    sycl::ext::intel::experimental::task_sequence<OpSqrt> task_a;

    for (int i = 0; i < len; i++){
      D3Vector item = InputPipeA::read();
      task_a.async(item, coef1);
    }

    for (int i = 0; i < len; i++){
      new_item[i] = task_a.get();
    }

    task_a.async(new_item, coef2);
    OutputPipeZ::write(task_a.get());
  }
};
```

</td>

<td>
<img src="assets/hardware_reuse_taskseq.svg" />
</td>

</tr>
</table>

## Sample Structure
The 3 different example designs in this sample perform similar operations. You may compare the C++ source files to see the code changes that are necessary to apply hardware reuse.

1. [Naive](naive/main.cpp) The component directly implements square root of dot product four times and each of them generates the hardware for its own use. 
2. [Looping](loop/main.cpp) Square root of dot product with same level of of data path are group together, and invoked in the 'for' loop. The hardware are shared by each invocation in the loop. Another square root of dot product, which is involed at the return point, generates its own hardware.
3. [Task sequence](task_sequence/main.cpp) Square root of dot product is invoked with a same task sequence object, both in the loop and at the return point. The hardware of it is shared by each invocation of the task.

## Build the `task_sequence_hardware_reuse` Tutorial
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
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

Use these commands to run the design, depending on your OS.

### On Linux* System

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake .. -DTYPE=<NAIVE/LOOP/TASK_SEQUENCE>
   ```
   >Use the appropriate `TYPE` parameter when running CMake to choose which design to compile:
   >| Example                                      | Directory             | Type (-DTYPE=) |
   >|----------------------------------------------|-----------------------|----------------|
   >| Naive                                        | naive/                | `NAIVE`        |
   >| Loop                                         | loop/                 | `LOOP`         |
   >| Task sequence                                | task-sequence/        | `TASK_SEQUENCE`|
   
   > **Note**: You can override the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number> -DTYPE=<NAIVE/LOOP/TASK_SEQUENCE>
   >  ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulates an FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the HTML optimization reports.
      ```
      make report
      ```
   3. Compile for simulation (fast compile time, targets simulator FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile with Quartus place and route (To get accurate area estimate, longer compile time).
      ```
      make fpga
      ```

### On Windows* System

1. Change to the sample directory.
2. Configure the build system for the Intel® Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" .. -DTYPE=<NAIVE/LOOP/TASK_SEQUENCE>
   ```
   >Use the appropriate `TYPE` parameter when running CMake to choose which design to compile:
   >| Example                                      | Directory             | Type (-DTYPE=) |
   >|----------------------------------------------|-----------------------|----------------|
   >| Naive                                        | naive/                | `NAIVE`        |
   >| Loop                                         | loop/                 | `LOOP`         |
   >| Task sequence                                | task-sequence/        | `TASK_SEQUENCE`|
   
   > **Note**: You can override the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number> -DTYPE=<NAIVE/LOOP/TASK_SEQUENCE>
   >  ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report. 
      ```
      nmake report
      ```
   3. Compile for simulation (fast compile time, targets simulator FPGA device).
      ```
      nmake fpga_sim
      ```
   4. Compile with Quartus place and route (To get accurate area estimate, longer compile time).
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Read the Reports
Locate `report.html` in the `naive.report.prj/reports/`, `naive_loop.report.prj/reports/` and `task_sequences.report.prj/reports/` directory.

Navigate to **System Resource Utilization Summary** (Summary > System Resource Utilization Summary) and compare the estimated area numbers in these report. You may found that area usage has been estimated to decrease as we moved towards task sequence implementation.

Locate `report.html` in the `naive.report.prj/reports/` directory and navigate to *System Viewer: Kernel system > VectorOpID > VectorOpID.B0 > Cluster 1* and *Cluster 2* to see FOUR `Floating-point sqrt` illustrated, representing FOUR replicate of hardware to be generated to complete the design.
<table>
<tr>
<td>
<img src="assets/naive_cluster1.svg" />
</td>
<td>
<img src="assets/naive_cluster2.svg" />
</td>
</tr>
</table>

Locate `report.html` in the `naive_loop.report.prj/reports/` directory.
Navigate to *System Viewer: Kernel system > VectorOpID > VectorOpID.B1 > Cluster 2* to see each loop invoke the same ONE `Floating-point sqrt`. 
Navigate to *System Viewer: Kernel system > VectorOpID > VectorOpID.B2 > Cluster 3* to find another `Floating-point sqrt`.
Total TWO replicate of hardware used in this design.
<table>
<tr>
<td>
<img src="assets/loop_cluster2.svg" />
</td>
<td>
<img src="assets/loop_cluster3.svg" />
</td>
</tr>
</table>

Locate `report.html` in the `task_sequences.report.prj/reports/` directory.
Navigate through the list in **System Viewer**, only ONE block of `Floating-point sqrt` can be found under *System Viewer: Kernel system > 3ull>) > 3ull>).B1 > Cluster 6*
<img src="assets/taskseq_cluster6.svg" />


## Run the Design

### On Linux
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./naive.fpga_emu
   ./naive_loop.fpga_emu
   ./task_sequences.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./naive.fpga_sim
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./naive_loop.fpga_sim
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./task_sequences.fpga_sim 
   ```

### On Windows
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   naive.fpga_emu.exe
   naive_loop.fpga_emu.exe
   task_sequences.fpga_emu.exe 
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   naive.fpga_sim.exe
   naive_loop.fpga_sim.exe
   task_sequences.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

## Example Output

### Naive
```
Running on device: Intel(R) FPGA Emulation Device
Processing vectors of size 3
PASSED
```
### Loop
```
Running on device: Intel(R) FPGA Emulation Device
Processing vectors of size 3
PASSED
```

### Task Sequence
```
Running on device: Intel(R) FPGA Emulation Device
Processing vectors of size 3
PASSED
```

## License
Code samples are licensed under the MIT license. See
[License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
