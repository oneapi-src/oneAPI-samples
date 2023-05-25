# `AC Int` Sample

This sample is an FPGA tutorial that demonstrates how to use the Algorithmic C (AC) integer data type `ac_int` and illustrates some recommended practices.

| Area                 | Description
|:--                   |:--
| What you will learn  | Using the `ac_int` data type for basic operations <br> Efficiently using the left shift operation <br> Setting and reading certain bits of an `ac_int` number
| Time to complete     | 20 minutes
| Category             | Concepts and Functionality

## Purpose

This FPGA tutorial shows how to use the `ac_int` data type with some simple examples.

This data type can be used in place of native integer types to generate area efficient and optimized designs for the FPGA. When you have a computation that does not require the full dynamic range of a 32-bit integer, you should replace your `int` variables with `ac_int` variables of the correct, reduced width. For example, if you know that a loop will iterate from 0 to 12 only 4 bits are required.

> **Note**: See the [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/optimize-your-design/resource-use/data-types-and-operations/var-prec-fp-sup/adv-disadv-ac-dt.html) to see advantages and limitations of `ac_int` data types.

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                  | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware            | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software            | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.

## Key Implementation Details

The sample illustrates the important concepts.

- The `ac_int` data type can be used to generate hardware for only as many bits as are needed by your application. Native integer types must generate hardware for only 8, 16, 32, or 64 bits.
- Shift operations in `ac_int` can be implemented more efficiently when the amount to shift by is stored in a minimally sized unsigned `ac_int`.
- The `ac_int` data type provides several useful operations, including reading and modifying certain bits in an `ac_int`.

### Simple Code Example

An `ac_int` number can be defined as follows:

```cpp
ac_int<W, S> a;
```

Here `W` is the width in bits and `S` is a bool indicating if the number is signed. Signed numbers use the most significant bit (MSB) to store the sign bit.

To use the `ac_int` type in your code, you must include the following header:

```cpp
#include <sycl/ext/intel/ac_types/ac_int.hpp>
```
Additionally, you must pass the  `-qactypes` option to the `icpx` command on Linux or the `/Qactypes` option to the `icx-cl` command on Windows when compiling your SYCL program in order to ensure that the headers are correctly included. In this tutorial, this is done in `src/CMakeLists.txt`.

### Basic Operations and Promotion Rules

When using `ac_int`, the results of addition, subtraction, multiplication, and division operations are automatically promoted to the number of bits needed to represent all possible results without overflowing. However, the data type you use to store the result may result in truncation.

For example, the addition of two 8-bit integers results in a 9-bit result to support overflow. Internally, the result will be 9-bit. However, if the user attempts to store the result in an 8-bit container, `ac_int` will let the user do this, which leads to the most significant bit being discarded. The responsibility lies on the user to use the correct data type.

These promotion rules are consistent across all architectures, so the behavior will be equivalent on x86 or on FPGA.

### Shift Operations

The behavior of shift operations of `ac_int` data types is slightly different from shift operations of native integer types. Some key points to remember are as follows:

- If the data type of the shift amount is not explicitly `unsigned` (either using `ac_int<N, false>` or using the `unsigned` keyword), then the compiler will generate a more complex shifter that allows negative shifts and positive shifts. A shift by a negative amount is equivalent to a positive shift in the opposite direction. Normally, you will not want to use negative shifting, so you should use an `unsigned` data type for the shift value to obtain a more resource efficient shifter.
- Shift values greater than the width of the data types are treated as a shift equal to the width of the data type.
- The shift operation can be done more efficiently by specifying the amount to shift with the smallest possible `ac_int`.

### Bit Select Operator

The bit select operator `[]` allows reading and modifying an individual bit in an `ac_int`.

*Note:* You must initialize an `ac_int` variable before accessing it using the bit select operator `[]`. Using the `[]` operator on an uninitialized `ac_int` variable is undefined behavior and can give you unexpected results. Assigning each bit explicitly using the `[]` operator does not count as initializing the `ac_int` variable.

### Bit Slice Operations

The slice read operation `slc` and the slice write operation `set_slc` allows reading and modifying a slice in an `ac_int`.

Slice read is provided with the template function `slc<int W>(int lsb)`. The two arguments are defined as:

- `W` is the bit length of the slice. It must be known at compile time.
- `lsb` is the index of the LSB of the slice being read.

Slice write is provided with the function `set_slc(int lsb, const ac_int<W, S> &slc)`. The two arguments are defined as:

- `lsb` is the index of the least significant bit (LSB) of the slice being written.
- `slc` is an `ac_int` slice that is to be written into the target `ac_int` starting at bit `lsb`. The bit length of slice is inferred from the width `W` of `slc`.

*Note:* An `ac_int` must be initialized before being accessed by bit slice operations `slc` and `set_slc`. Using the `slc` and `set_slc` functions on an uninitialized `ac_int` variable is undefined behavior and can give you unexpected results.

### Understanding the Tutorial Design

This tutorial consists of five kernels:

Kernel `BasicOpsInt` contains native `int` type addition, multiplication, and division operations, while kernel `BasicOpsAcInt` contains `ac_int` type addition, multiplication, and division operations. By comparing these two kernels, you will find reduced width `ac_int` generates hardware that is more area efficient than native `int`.

Kernel `ShiftOps` contains an `ac_int` left-shifter and an `ac_int` right-shifter, and the data type of the shift amount is a large width signed `ac_int`. In contrast, kernel `EfficientShiftOps` also contains an `ac_int` left-shifter and an `ac_int` right-shifter, but the data type of the shift amount is a reduced width unsigned `ac_int`. By comparing these two kernels, you will find shift operations of `ac_int` can generate more efficient hardware if the amount to shift by is stored in a minimally sized unsigned `ac_int`.

Kernel `BitOps` demonstrates bit operations with bit select operator `[]` and bit slice operations `slc` and `set_slc`.

## Build the `AC Int` Tutorial

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

Locate `report.html` in the `ac_int_report.prj/reports/` directory.

On the main report page, scroll down to the section titled *Compile Estimated Kernel Resource Utilization Summary*. You can see the overall resource usage of kernel `BasicOpsAcInt` is less than kernel `BasicOpsInt`. Navigate to *Area Analysis of System* (*Area Analysis* > *Area Analysis of System*), you can find resource usage information of the individual addition, multiplication, and division operations, and you can verify that each individual operation consumes fewer resources in kernel `BasicOpsAcInt` than in kernel `BasicOpsInt`.

Navigate to *System Viewer* (*Views* > *System Viewer*) and find the cluster in kernel `ShiftOps` that contains the left-shifter node (`<<`) and the right-shifter node (`>>`). Similarly, locate the cluster that contains the left-shifter node and the right-shifter node in kernel `EfficientShiftOps`. Observe that the compiler generates an additional shifter in kernel `ShiftOps` to deal with the signedness of the shift amount `b`. You can verify that kernel `EfficientShiftOps` consumes fewer resources than kernel `ShiftOps` in *Compile Estimated Kernel Resource Utilization Summary* on the main report page and *Area Analysis of System*.

## Run the `AC Int` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./ac_int.fpga_emu
   ```
2. Run the sample of the FPGA simulator device (the kernel executes on the CPU).
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./ac_int.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./ac_int.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ac_int.fpga_emu.exe
   ```
2. Run the sample of the FPGA simulator device (the kernel executes on the CPU).
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   ac_int.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ac_int.fpga.exe
   ```

## Example Output

You will see the device used. If successful, the program displays output similar to the following:

```
PASSED: all kernel results are correct.
```

### Understand the Results

Using `ac_int` can help minimize the generated hardware and achieve the same numerical result as native integer types. This approach is useful when the logic does not need to use all the bits provided by the native integer type.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).