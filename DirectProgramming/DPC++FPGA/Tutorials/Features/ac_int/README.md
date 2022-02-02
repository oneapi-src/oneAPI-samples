# Using the Algorithmic C Integer Data Type `ac_int`

This FPGA tutorial demonstrates how to use the Algorithmic C (AC) data type `ac_int` and some best practices.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Using the `ac_int` data type for basic operations <br> Efficiently using the left shift operation <br> Setting and reading certain bits of an `ac_int` number
| Time to complete                  | 20 minutes



## Purpose

This FPGA tutorial shows how to use the `ac_int` data type with some simple examples.

This data type can be used in place of native integer types to generate area efficient and optimized designs for the FPGA. When you have a computation that does not require the full dynamic range of a 32-bit integer, you should replace your `int` variables with `ac_int` variables of the correct, reduced width. For example, if you know that a loop will iterate from 0 to 12, only 4 bits are required.

Please refer to the [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/optimize-your-design/resource-use/data-types-and-operations/var-prec-fp-sup/adv-disadv-ac-dt.html) to see advantages and limitations of `ac_int` data types.

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
Additionally, you must pass the flag `-qactypes` on Linux or `/Qactypes` on Windows to the `dpcpp` command when compiling your SYCL program in order to ensure that the headers are correctly included. In this tutorial, this is done in `src/CMakeLists.txt`.

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

Kernel `ShiftOp` contains an `ac_int` left shifter and the data type of the shift amount is a large width signed `ac_int`. In contrast, kernel `EfficientShiftOp` also contains an `ac_int` left shifter, but the data type of the shift amount is a reduced width unsigned `ac_int`. By comparing these two kernels, you will find shift operations of `ac_int` can generate more efficient hardware if the amount to shift by is stored in a minimally sized unsigned `ac_int`.

Kernel `BitOps` demonstrates bit operations with bit select operator `[]` and bit slice operations `slc` and `set_slc`.

## Key Concepts
* The `ac_int` data type can be used to generate hardware for only as many bits as are needed by your application. Native integer types must generate hardware for only 8, 16, 32, or 64 bits.
* Shift operations in `ac_int` can be implemented more efficiently when the amount to shift by is stored in a minimally sized unsigned `ac_int`.
* The `ac_int` data type provides several useful operations, including reading and modifying certain bits in an `ac_int`.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

## Building the `ac_int` Tutorial

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

1. Install the design in `build` directory from the design directory by running `cmake`:

   ```bash
   mkdir build
   cd build
   ```

   If you are compiling for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:

   ```bash
   cmake ..
   ```

   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```bash
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```bash
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design using the generated `Makefile`. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulates an FPGA device) using:

     ```bash
     make fpga_emu
     ```

   * Generate HTML optimization reports using:

     ```bash
     make report
     ```

   * Compile and run on FPGA hardware (longer compile time, targets an FPGA device) using:

     ```bash
     make fpga
     ```

3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/ac_int.fpga.tar.gz" download>here</a>.

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

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.
 
### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*).
For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports

Locate `report.html` in the `ac_int_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

On the main report page, scroll down to the section titled *Compile Estimated Kernel Resource Utilization Summary*. You can see the overall resource usage of kernel `BasicOpsAcInt` is less than kernel `BasicOpsInt`. Navigate to *Area Analysis of System* (*Area Analysis* > *Area Analysis of System*), you can find resource usage information of the individual addition, multiplication, and division operations, and you can verify each individual operation consumes fewer resources in kernel `BasicOpsAcInt` than in kernel `BasicOpsInt`.

Navigate to *System Viewer* (*Views* > *System Viewer*) and find the cluster in kernel `ShiftOp` that contains the left-shifter node (`<<`). Similarly, locate the cluster that contains the left-shifter node in kernel `EfficientShiftOp`. Observe that the compiler generates an additional shifter in kernel `ShiftOp` to deal with the signedness of the shift amount `b`. You can verify that kernel `EfficientShiftOp` consumes fewer resources than kernel `ShiftOp` in *Compile Estimated Kernel Resource Utilization Summary* on the main report page and *Area Analysis of System*.

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):

   ```bash
   ./ac_int.fpga_emu     (Linux)
   ac_int.fpga_emu.exe   (Windows)
   ```

2. Run the sample on the FPGA device

   ```bash
   ./ac_int.fpga         (Linux)
   ```

### Example of Output

```txt
PASSED: all kernel results are correct.
```

### Discussion

`ac_int` can help minimize the generated hardware and achieve the same numerical result as native integer types. This can be very useful when the logic does not need to utilize all the bits provided by the native integer type.
