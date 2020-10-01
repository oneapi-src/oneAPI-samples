# Compiling DPC++ for FPGA
This FPGA tutorial introduces how to compile DPC++ for FPGA through a simple vector addition example.

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | How and why compiling DPC++ to FPGA differs from CPU or GPU <br> FPGA device image types and when to use them <br> The compile flags used to target FPGA
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose
Field-programmable gate arrays (FPGAs) are configurable integrated circuits that can be programmed to implement arbitrary circuit topologies. Classified as *spatial* compute architectures, FPGAs differ significantly from fixed Instruction Set Architecture (ISA) devices like CPUs and GPUs, and offer a different set of optimization trade-offs from these traditional accelerator devices.

While DPC++ can be compiled for CPU, GPU or for FPGA, the process for compiling to FPGA is somewhat different than for CPU or GPU. This tutorial motivates these differences and explains how to compile a "Hello World"-style vector addition kernel for FPGA.

### Why is FPGA compilation different?
FPGAs differ from CPUs and GPUs in many interesting ways. However, in the scope of this tutorial, there is only one difference that matters: compared to CPU or GPU, generating a device image for FPGA hardware is a computationally intensive and time-consuming process. It is normal for an FPGA compile to take several hours to complete.

For this reason, only ahead-of-time (or "offline") kernel compilation mode is supported for FPGA. The long compile time for FPGA hardware makes just-in-time (or "online") compilation impractical.

Long compile times are detrimental to developer productivity. The Intel® oneAPI DPC++ Compiler provides several mechanisms that enable DPC++ developers targeting FPGA to iterate quickly on their designs. By circumventing the time-consuming process of full FPGA compilation wherever possible, DPC++ FPGA developers can enjoy the fast compile times familiar to CPU and GPU developers.


### Three types of DPC++ FPGA compilation
The three types of FPGA compilation are summarized in the table below.

| Device Image Type    | Time to Compile | Description
---                    |---              |---
| FPGA Emulator        | seconds         | The FPGA device code is compiled to the CPU. <br> This is used to verify the code's functional correctness.
| Optimization Report  | minutes         | The FPGA device code is partially compiled for hardware. <br> The compiler generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization.
| FPGA Hardware        | hours           | Generates the real FPGA bitstream to execute on the target FPGA platform

The typical FPGA DPC++ development workflow is to iterate in each of these stages, refining the code using the feedback provided by that stage. Intel® recommends relying on emulation and the optimization report whenever possible.

Compiling for FPGA emulation or to generate the FPGA optimization report requires only the Intel® oneAPI DPC++ Compiler (part of the Intel® oneAPI Base Toolkit). An FPGA hardware compile requires the Intel® FPGA Add-On for oneAPI Base Toolkit.


#### FPGA Emulator

The FPGA emulator is the fastest method to verify the correctness of your code. The FPGA emulator executes DPC++ device code on the CPU. The emulator is similar to the SYCL* host device, but unlike the host device the FPGA emulator device supports FPGA extensions such as FPGA pipes and `fpga_reg`.

There are two important caveats to remember when using the FPGA emulator.
*  **Performance is not representative.** It is not meaningful to evaluate performance on the FPGA emulator, as it is not representative of the behavior of the FPGA device. For example, an optimization that yields a 100x performance improvement on the FPGA may show no impact on the emulator performance, or it may show an unrelated increase or decrease.
* **Undefined behavior may differ.** If your code produces different results when compiled for the FPGA emulator versus FPGA hardware, it is likely that your code is exercising undefined behavior. By definition, undefined behavior is not specified by the language specification, and may manifest differently on different targets.

#### Optimization Report
An full FPGA compilation occurs in two stages:
1. **FPGA early image:** The DPC++ device code is optimized and converted into an FPGA design specified in Verilog RTL (a low-level, native entry language for FPGAs). This intermediate compilation result is the FPGA early device image, which is *not* executable. This FPGA early image compilation process takes minutes.
2. **FPGA hardware image:** The Verilog RTL specifying the design's circuit topology is mapped onto the FPGA's sea of primitive hardware resources by the Intel® Quartus® Prime software.  Intel® Quartus® Prime is included in the Intel® FPGA Add-On, which is required for this compilation stage. The result is an FPGA hardware binary (also referred to as a bitstream). This compilation process takes hours.

Optimization reports are generated after both stages. The optimization report generated after the FPGA early device image, sometimes called the "static report", contains significant information about how the compiler has transformed your DPC++ device code into an FPGA design. The report contains visualizations of structures generated on the FPGA, performance and expected performance bottleneck information, and estimated resource utilization.

The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/analyze-your-design.html) contains a chapter on how to analyze the reports generated after the FPGA early image and FPGA image.

#### FPGA Hardware
This is a full compile through to the FPGA hardware image. You can target the Intel® PAC with Intel Arria® 10 GX FPGA, the Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA, or a custom board.

### Device Selectors
The following code  snippet demonstrates how you can specify the target device in your source code. The selector is used to specify the target device at runtime.

```c++
// FPGA device selectors are defined in this utility header
#include <CL/sycl/INTEL/fpga_extensions.hpp>

int main() {
  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  queue q(device_selector);
  ...
}
```
Notice that the FPGA emulator and the FPGA are are different target devices. It is recommended to use a preprocessor define to choose between the emulator and FPGA selectors.  This makes it easy to switch between targets using only command-line flags. Since the FPGA only supports ahead-of-time compilation, dynamic selectors (such as the default_selector) are less useful than explicit selectors when targeting FPGA.


### Compiler Flags
Here is a cheat sheet of the DPC++ compiler commands to compile for the FPGA emulator, to generate the FPGA early image optimization reports, and to compile for FPGA hardware.
```
# FPGA emulator
dpcpp -fintelfpga -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu

# Optimization report (default board)
dpcpp -fintelfpga -Xshardware -fsycl-link fpga_compile.cpp -o fpga_compile_report.a
# Optimization report (explicit board)
dpcpp -fintelfpga -Xshardware -fsycl-link -Xsboard=intel_s10sx_pac:pac_s10 fpga_compile.cpp -o fpga_compile_report.a

# FPGA hardware (default board)
dpcpp -fintelfpga -Xshardware fpga_compile.cpp -o fpga_compile.fpga
# FPGA hardware (explicit board)
dpcpp -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 fpga_compile.cpp -o fpga_compile.fpga
```

The compiler flags used to achieve this are explained below.
| Flag              | Explanation
---                 |---
| `-fintelfpga`     | Perform ahead-of-time compilation for FPGA.
| `-DFPGA_EMULATOR` | Adds a preprocessor define (see code snippet above).
| `-Xshardware`     | `-Xs` is used to pass arguments to the FPGA backend. <br> Since emulator is the default FPGA target, you must pass `Xshardware` to instruct the compiler to target FPGA hardware.
| `-Xsboard`        | Optional argument to specify the FPGA board target. <br> If omitted, a default FPGA board is chosen.
| `-fsycl-link`     | This is synonymous with `-fsycl-link=early`. <br> It instructs the compile to stop after creating the FPGA early image (and associated optimization report).

Notice that whether you are targeting the FPGA emulator or FPGA hardware must be specified twice: through compiler flags for the ahead-of-time compilation, and through the device selector for the runtime.


## Key Concepts
* How and why compiling DPC++ to FPGA differs from CPU or GPU
* FPGA device image types and when to use them
* The compile flags used to target FPGA

## License
This code sample is licensed under MIT license.


## Building the `fpga_compile` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile or fpga_runtime) as well as whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/get-started/base-toolkit/](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

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
   Alternatively, to compile for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for [emulation](#fpga-emulator) (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Generate the [optimization report](#optimization-report):
     ```
     make report
     ```
   * Compile for [FPGA hardware](#fpga-hardware) (longer compile time, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/pipe_array.fpga.tar.gz" download>here</a>.
### On a Windows* System
Note: `cmake` is not yet supported on Windows. A build.ninja file is provided instead. 

1. Enter the source file directory.
   ```
   cd src
   ```

2. Compile the design. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
      ```
      ninja fpga_emu
      ```
      **NOTE:** For the FPGA emulator target, the device link method is used. 
   * Generate the optimization report:

     ```
     ninja report
     ```
     If you are targeting Intel® PAC with Intel Stratix® 10 SX FPGA, instead use:
     ```
     ninja report_s10_pac
     ```     
   * Compiling for FPGA hardware is not yet supported on Windows.
 
### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)


## Examining the Reports
Locate `report.html` in the `fpga_compile_report.prj/reports/` or `fpga_compile_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Browse the reports that were generated for the `VectorAdd` kernel's FPGA early image. You may also wish to examine the reports generated by the full FPGA hardware compile and compare their contents.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./fpga_compile.fpga_emu     (Linux)
     fpga_compile.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./fpga_compile.fpga         (Linux)
     ```

### Example of Output
```
PASSED: results are correct
```
