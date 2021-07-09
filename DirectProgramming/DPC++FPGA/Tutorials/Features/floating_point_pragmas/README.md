# Contraction and Reassociation on Floating Point Numbers
This FPGA tutorial explains how to use the `fp reassociate` and `fp contract` pragmas for floating point numbers. These pragmas allow the compiler to skip the rounding steps or change the order of certain floating point operations so that they map more efficiently to hardware. The impact is that the results may be altered slightly due to rounding that can occur after each floating point operation.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | The basic usage of the `fp contract(fast\|off)` and `fp reassociate(on\|off)` pragmas <br> How the `fp contract(fast\|off)` and `fp reassociate(on\|off)` pragmas affect resource use and latency <br> How to apply the `fp contract(fast\|off)` and `fp reassociate(on\|off)` pragmas in your program
| Time to complete                  | 20 minutes



## Purpose
This FPGA tutorial demonstrates an example of using `fp contract(fast|off)` pragma to:
+ Skip intermediate rounding and conversion between double precision arithmetic operations with acceptable error
+ Reduce area and improve latency

and `fp reassociate(on|off)` pragma to:
+ Relaxing the order of floating point arithmetic operations
+ Reduce area and improve latency

#### Example: Turn On Fp Contract
```c++
#pragma fp contract(fast)
a = b * c + d;
```

The `fp contract(fast)` pragma allows the compiler to skip rounding steps on these double precision operations, which in turn allows more efficient use of FPGA floating point math resources and thus a reduction in area and latency.

```c++
#pragma fp reassociate(on)
a = b + c + d + e;
```

By relaxing the order of the additions using `fp reassociate(on)`, the compiler is able to group these four additions in a way that maps to the FPGA hardware more efficiently, and thus saves area and reduces latency.

## Key Concepts
* The basic usage of the `fp contract(fast|off)` and `fp reassociate(on|off)` pragmas
* How the `fp contract(fast|off)` and `fp reassociate(on|off)` pragmas affect resource use and latency
* How to apply the `fp contract(fast|off)` and `fp reassociate(on|off)` pragmas in your program

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `floating_point_pragmas` Tutorial

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
3. (Optional) As the FPGA hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/fp_pragmas.fpga.tar.gz" download>here</a>.

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

Locate `report.html` files in either:

* **Report-only compile**:  `fp_pragmas_report.prj`
* **FPGA hardware compile**: `fp_pragmas.prj`

Open the reports in Google Chrome*, Mozilla Firefox*, Microsoft Edge*, or Microsoft Internet Explorer*.

On the main report page, scroll down to the section titled "Estimated Resource Usage". Each kernel name represents the pragma usage. e.g., `ContractFastKernel` sets the `fp contract(fast|off)` pragma to `fast` mode. You can verify that the number of ALMs used for kernels with `fp contract(fast)` or `fp reassociate(on)` is fewer than the kernels with `fp contract(off)` or `fp reassociate(off)` respectively.

In the "Loop Analysis" section under the tab titled "Throughput Analysis", there is a panel on the left containing all the kernels. the loop information of each kernel can be seen by expanding them. You can verify that the latencies for kernels with `fp contract(fast)` or `fp reassociate(on)` are smaller than those of the kernels with `fp contract(off)` or `fp reassociate(off)` respectively.

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):

   ```bash
   ./fp_pragmas.fpga_emu    # Linux
   fp_pragmas.fpga_emu.exe  # Windows
   ```

2. Run the sample on the FPGA device

   ```bash
   ./fp_pragmas.fpga        # Linux

### Example of Output

```txt
Contract enabled: 
    kernel result: -4.98704e+08, correct result: -4.98704e+08
    delta: 2.38419e-07, expected delta: less than 0.000001
Contract disabled: 
    kernel result: -4.98704e+08, correct result: -4.98704e+08
    delta: 0, expected delta: 0
Reassociate enabled: 
    kernel result: -4.98704e+08, correct result: -4.98704e+08
    delta: 1.19209e-07, expected delta: less than 0.000001
Reassociate disabled: 
    kernel result: -4.98704e+08, correct result: -4.98704e+08
    delta: 0, expected delta: 0
Test Passed
```

### Discussion of Results

You can see the reduction of resource use in kernel ContractFastKernel comparing with ContractOffKernel. You will also note the decrease in resource use in kernel ReassociateOnKernel comparing with ReassociateOffKernel.
