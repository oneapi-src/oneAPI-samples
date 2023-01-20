# Compiling SYCL* for FPGA
This FPGA tutorial introduces how to compile SYCL*-compliant code for FPGA through a simple vector addition example. If you are new to SYCL* for FPGA, start with this sample.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Agilex™, Arria® 10, and Stratix® 10 FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | How and why compiling SYCL* code for FPGA differs from CPU or GPU <br> FPGA device image types and when to use them <br> The compile options used to target FPGA
| Time to complete                  | 15 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 1 sample that helps you getting started.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")
   
   tier1 --> tier2 --> tier3 --> tier4
   
   style tier1 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/DPC++FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/DPC++FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/DPC++FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/DPC++FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/DPC++FPGA/README.md#documentation), etc.

## Purpose
Field-programmable gate arrays (FPGAs) are configurable integrated circuits that can be programmed to implement arbitrary circuit topologies. Classified as *spatial* compute architectures, FPGAs differ significantly from fixed Instruction Set Architecture (ISA) devices like CPUs and GPUs. FPGAs offer a different set of optimization trade-offs from these traditional accelerator devices.

While SYCL* code can be compiled for CPU, GPU, or FPGA, compiling to FPGA is somewhat different. This tutorial explains these differences and shows how to compile a "Hello World" style vector addition kernel for FPGA, following the recommended workflow.

### Why is compilation different for FPGA?
FPGAs differ from CPUs and GPUs in many interesting ways. However, in this tutorial's scope, there is only one difference that matters: compared to CPU or GPU, generating a device image for FPGA hardware is a computationally intensive and time-consuming process. It is usual for an FPGA compile to take several hours to complete.

For this reason, only ahead-of-time (or "offline") kernel compilation mode is supported for FPGA. The long compile time for FPGA hardware makes just-in-time (or "online") compilation impractical.

Long compile times are detrimental to developer productivity. The Intel® oneAPI DPC++/C++ Compiler provides several mechanisms that enable developers targeting FPGA to iterate quickly on their designs. By circumventing the time-consuming process of full FPGA compilation wherever possible, SYCL for FPGA developers can enjoy the fast compile times familiar to CPU and GPU developers.


### Three types of SYCL for FPGA compilation
The three types of FPGA compilation are summarized in the table below.

| Device Image Type    | Time to Compile | Description
---                    |---              |---
| FPGA Emulator        | seconds         | The FPGA device code is compiled to the CPU. <br> This is used to verify the code's functional correctness.
| Optimization Report  | minutes         | The FPGA device code is partially compiled for hardware. <br> The compiler generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization.
| FPGA Hardware        | hours           | Runs Intel® Quartus® to get accurate resource usage and fmax estimates. If a BSP is targeted, generates the real FPGA bitstream to execute on the target FPGA platform

The typical FPGA development workflow is to iterate in each of these stages, refining the code using the feedback provided by that stage. Intel® recommends relying on emulation and the optimization report whenever possible.

Compiling for FPGA emulation or generating the FPGA optimization report requires only the Intel® oneAPI DPC++/C++ Compiler (part of the Intel® oneAPI Base Toolkit).

#### FPGA Emulator

The FPGA emulator is the fastest method to verify the correctness of your code. The FPGA emulator executes the SYCL* device code on the CPU. The emulator is similar to the SYCL* host device, but unlike the host device, the FPGA emulator device supports FPGA extensions such as FPGA pipes and `fpga_reg`.

#### FPGA Simulator

The FPGA simulator is the fastest method to verify the correctness of the gerenated RTL. The FPGA simulator executes the SYCL* device code in an RTL simulator (e.g. Questa*). The host code still runs on the CPU as it would when targetting an FPGA. When using this flow, the generated exectuable will launch the simulator and inject the obtained results in the host execution.

There are two important caveats to remember when using the FPGA emulator and the FPGA simulator.
*  **Performance is not representative.** _Never_ draw inferences about FPGA performance from the FPGA emulator. The FPGA emulator's timing behavior is uncorrelated to that of the physical FPGA hardware. For example, an optimization that yields a 100x performance improvement on the FPGA may show no impact on the emulator performance. It may show an unrelated increase or even a decrease.
* **Undefined behavior may differ.** If your code produces different results when compiled for the FPGA emulator versus FPGA hardware, your code most likely exercises undefined behavior. By definition, undefined behavior is not specified by the language specification and may manifest differently on different targets.

#### Optimization Report
A full FPGA compilation occurs in two stages:
1. **FPGA early image:** The SYCL device code is optimized and converted into an FPGA design specified in Verilog RTL (a low-level, native entry language for FPGAs). This intermediate compilation result is the FPGA early device image, which is *not* executable. This FPGA early image compilation process takes minutes.
2. **FPGA hardware image:** The Verilog RTL specifying the design's circuit topology is mapped onto the FPGA's sea of primitive hardware resources by the Intel® Quartus® Prime software.  Intel® Quartus® Prime is included in the Intel® FPGA Add-On, which is required for this compilation stage. The result is an FPGA hardware binary (also referred to as a bitstream). This compilation process takes hours.

Optimization reports are generated after both stages. The optimization report generated after the FPGA early device image, sometimes called the "static report," contains significant information about how the compiler has transformed your device code into an FPGA design. The report includes visualizations of structures generated on the FPGA, performance and expected performance bottleneck information, and estimated resource utilization.

The [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/analyze-your-design.html) contains a chapter on how to analyze the reports generated after the FPGA early image and FPGA image.

#### FPGA Hardware
This is a full compile through to the FPGA hardware image. 
You can target an FPGA family/part number to get accurate resource usage and fmax estimates.
You can also target a device with a BSP (e.g. for the Intel® PAC with Intel Arria® 10 GX FPGA: intel_a10gx_pac:pac_a10) to get an executable that can be directly executed.

### Device Selectors
The following code snippet demonstrates how you can specify the target device in your source code. The selector is used to specify the target device at runtime.

```c++
// FPGA device selectors are defined in this utility header
#include <sycl/ext/intel/fpga_extensions.hpp>

int main() {
  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA simulator
  //  - the FPGA device (a real FPGA)
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  queue q(selector);
  ...
}
```
Notice that the FPGA emulator, FPGA simulator and the FPGA are different target devices. It is recommended to use a preprocessor define to choose between the different selectors. This makes it easy to switch between targets using only command-line options. Since the FPGA only supports ahead-of-time compilation, dynamic selectors (such as the default_selector) are less useful than explicit selectors when targeting FPGA.

### Compiler Options
This section includes a helpful list of commands and options to compile this design for the FPGA emulator, generate the FPGA early image optimization reports, and compile for FPGA hardware.

**NOTE:** In this sample, the compiler is refered to as `icpx`. On Windows, you should use `icx-cl`.

**FPGA emulator**

`icpx -fsycl -fintelfpga -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu`

**FPGA simulator**

`icpx -fsycl -fintelfpga -Xssimulation -DFPGA_SIMULATOR fpga_compile.cpp -o fpga_compile.fpga_sim`

**Optimization report (default FPGA device)**

`icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -fsycl-link=early fpga_compile.cpp -o fpga_compile_report.a`

**Optimization report (explicit FPGA device)**

`icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -fsycl-link=early -Xstarget=intel_s10sx_pac:pac_s10 fpga_compile.cpp -o fpga_compile_report.a`

**FPGA hardware (default FPGA device)**

`icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware fpga_compile.cpp -o fpga_compile.fpga`

**FPGA hardware (explicit FPGA device)**

`icpx -fsycl -fintelfpga -DFPGA_HARDWARE -Xshardware -Xstarget=intel_s10sx_pac:pac_s10 fpga_compile.cpp -o fpga_compile.fpga`

The compiler options used are explained in the table.
| Flag               | Explanation
|:---                |:---
| `-fsycl`           | Instructs the compiler that the code is written in the SYCL language
| `-fintelfpga`      | Perform ahead-of-time compilation for FPGA.
| `-DFPGA_EMULATOR`  | Adds a preprocessor define that invokes the emulator device selector in this sample (see code snippet above).
| `-DFPGA_SIMULATOR` | Adds a preprocessor define that invokes the simulator device selector in this sample (see code snippet above).
| `-DFPGA_HARDWARE`  | Adds a preprocessor define that invokes the FPGA hardware device selector in this sample (see code snippet above).
| `-Xshardware`      | `-Xs` is used to pass arguments to the FPGA backend. <br> Since the emulator is the default FPGA target, you must pass `Xshardware` to instruct the compiler to target FPGA hardware.
| `-Xstarget`        | Optional argument to specify the FPGA target. <br> If omitted, a default FPGA board is chosen.
| `-fsycl-link=early`| Instructs the compiler to stop after creating the FPGA early image (and associated optimization report).

Notice that whether you target the FPGA emulator, FPGA simulator or FPGA hardware must be specified twice: through compiler options for the ahead-of-time compilation and through the runtime device selector.

## Key Concepts
* How and why compiling SYCL*-compliant code to FPGA differs from CPU or GPU
* FPGA device image types and when to use them
* The compile options used to target FPGA

## Building the `fpga_compile` Tutorial

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
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex™ device family), run `cmake` using the command:
  ```
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

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

  * Compile for [emulation](#fpga-emulator) (compiles quickly, targets emulated FPGA device):
    ```
    make fpga_emu
    ```
  * Compile for [simulation](#fpga-simulator) (fast compile time, targets simulator FPGA device):
    ```
    make fpga_sim
    ```
  * Generate the [optimization report](#optimization-report):
    ```
    make report
    ```
  * Compile for [FPGA hardware](#fpga-hardware) (takes longer to compile, targets FPGA device):
    ```
    make fpga
    ```

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex™ device family), run `cmake` using the command:
  ```
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

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

  * Compile for [emulation](#fpga-emulator) (compiles quickly, targets emulated FPGA device):
    ```
    nmake fpga_emu
    ```
  * Compile for [simulation](#fpga-simulator) (fast compile time, targets simulator FPGA device):
    ```
    nmake fpga_sim
    ```
  * Generate the [optimization report](#optimization-report):
    ```
    nmake report
    ```
  * Compile for [FPGA hardware](#fpga-hardware) (takes longer to compile, targets FPGA device):
    ```
    nmake fpga
    ```

## Examining the Reports
Locate `report.html` in the `fpga_compile_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Browse the reports that were generated for the `VectorAdd` kernel's FPGA early image. You may also wish to examine the reports generated by the full FPGA hardware compile and compare their contents.

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
  ```
  ./fpga_compile.fpga_emu     (Linux)
  fpga_compile.fpga_emu.exe   (Windows)
  ```
2. Run the sample on the FPGA simulator device (the kernel executes in the simulator):
  * On Linux
    ```bash
    CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./fpga_compile.fpga_sim
    ```
  * On Windows
    ```bash
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
    fpga_compile.fpga_sim.exe
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
    ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`):
  ```
  ./fpga_compile.fpga         (Linux)
  fpga_compile.fpga.exe       (Windows)
  ```

### Example of Output
```
PASSED: results are correct
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
