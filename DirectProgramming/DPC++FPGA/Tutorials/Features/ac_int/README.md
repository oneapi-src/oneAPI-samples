# Using the Algorithmic C Integer Data-type 'ac_int'

This FPGA tutorial demonstrates how to use the Algorithmic C (AC) Data-type `ac_int` and some best practices.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Using the `ac_int` data-type for basic operations <br> Efficiently using the left shift operation <br> Setting and reading certain bits of an `ac_int` number
| Time to complete                  | 20 minutes



## Purpose

This FPGA tutorial shows how to use the `ac_int` type with some simple examples.

This data-type can be used in place of native integer types to generate area efficient and optimized designs for the FPGA. For example, operations which do not utilize all of the bits the native integer types are good candidates for replacement with `ac_int` type.


### Simple Code Example

An `ac_int` number can be defined as follows:
```cpp
ac_int<W, S> a;
```
Here W is the width and S is the sign of the number. Signed numbers use one of the W bits to store the sign information.

To use this type in your code, you must include the following header:

```cpp
#include <sycl/ext/intel/ac_types/ac_int.hpp>
```
Additionally, you must use the flag `-qactypes` in order to ensure that the headers are correctly included.

For convenience, the following are predefined under the `ac_intN` namespace:
```
ac_int<N, true>  are type defined as intN up to 63.
ac_int<N, false> are type defined as uintN up to 63.
```

For example, a 14 bit signed `ac_int` can be defined by using
```cpp
ac_intN::int14 a;
```

### Understanding the Tutorial Design

The tutorial consists of several functions, each of which contains a SYCL kernel that demonstrates a specific operation. The operations we will see are:
* Addition
* Division
* Multiplication
* Left shift
* Setting a bit of an `ac_int` number
* Reading a bit of an `ac_int` number

#### Basic Operations and Promotion Rules

When using `ac_int`, we can write Addition, Division, Multiplication operations to use precisely as many bits as are needed to store the results. This is demonstrated by the kernels `Add`, `Div` and `Mult`.

`ac_int` automatically promotes the result of all operations to the number of bits needed to represent all possible results without overflowing. For example, the addition of two 8-bit integers results in a 9-bit result to support overflow.

However, if the user attempts to store the result in an 8-bit container, `ac_int` will let the user do this, but this leads to the discard of the extra carry bit. The responsibility lies on the user to use the correct datatype.

These promotions rules are consistent across all architectures so the behavior should be equivalent on x86 or on FPGA.

#### Shift Operation

The behavior of a shift operation with an `ac_int` is slightly different from its behavior with native integer types. For full details, see the `ac_int` documentation in the file `ac_data_types_ref.pdf`. Some key points to remember are as follows:
  - If the datatype of the shift amount is not explicitly `unsigned` (either using `ac_int<N, *false*>` or using the `unsigned` keyword), then the compiler will generate a more complex shifter that allows negative shifts and positive shifts. A right-shift by a negative amount is equivalent to a positive left-shift.
  - Normally, you will not want to enable negative shifting, so you should use an `unsigned` datatype for the shift value to obtain a more resource efficient design.
  - Shift values greater than the width of the data types are treated as a shift equal to the width of the datatype.
  - The shift operation can be done more efficiently by specifying the amount to shift with the smallest possible `ac_int`.

For example, in the tutorial, two kernels perform the left shift operation: `ShiftLeft` and `EfficientShiftLeft`. Both operate on an 14 bits wide `ac_int`. The former stores the shift amount in an `ac_int` which is 14 bits wide and the latter stores it in an `ac_int` which is 4 bits wide. The latter will generate simpler hardware.

#### Bit Slice Operations

The kernels `GetBitSlice` and `SetBitSlice` show how to read from and write to specific bits of an `ac_int` number. Note that only static bit widths are supported with such "slice" operations.

For detailed documentation on the `set_slc` and `slc` APIs please see the file `ac_data_types_ref.pdf`

## Key Concepts
* The `ac_int` data-type can be used to generate hardware for only as many bits as is needed by the operation as compared to native integer types which generate hardware for the entire type width.
* The left shift operation on `ac_int` can be implemented more efficiently when the amount to shift with is stored in a minimally sized ac_int``.
* The `ac_int` data-type offers functions for several useful operations including reading and writing of certain bits of an `ac_int` number. This can be very useful in creating bit masks.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

## Building the `ac_int` Tutorial

### Include Files

The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

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

Navigate to the *System Viewer* report (*Views* > *System Viewer*) and step through the clusters generated for `ShiftLeft` by clicking on the cluster entires on the left hand side pane under `ShiftLeft` until you find the one that contains the left shift node (`<<`). Similarly locate the cluster containing the left shift node for `EfficientShiftLeft`. Observe that the compiler needs to generate extra logic to deal with the signedness of the b operand for the `ShiftLeft` kernel and hence generates more hardware than for the `EfficientShiftLeft` kernel.

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):

   ```bash
   ./ac_int.fpga_emu    # Linux
   ac_int.fpga_emu.exe  # Windows
   ```

2. Run the sample on the FPGA device

   ```bash
   ./ac_int.fpga             # Linux
   ```

### Example of Output

```txt
Arithmetic Operations:
ac_int: +1383 + +966 = +2349
int:    1383 + 966 = 2349
ac_int: +6249 * +966 = +6036534
int:    6249 * 966 = 6036534
ac_int: +2163 / +43 = +50
int:    2163 / 43 = 50

Bitwise Operations:
ac_int: +7423 << +2 = -3076
int:    7423 << 2 = -3076
ac_int: +6380 << 1 = -3624
int:    6380 << 1 = -3624
(+7373).slc<4>(5) = 6
Running these two ops on +7373
        (+7373).set_slc(6, 10) = +7808
        a[3] = 0; a[2] = 0; a[1] = 0; a[0] = 0;
        Result = +7808
PASSED
```

### Discussion of Results

`ac_int` can help minimize the generated hardware and achieve the same numerical result as standard integer types. This can be very useful when the logic does not need to utilize all of the bits provided by the standard integer type.
