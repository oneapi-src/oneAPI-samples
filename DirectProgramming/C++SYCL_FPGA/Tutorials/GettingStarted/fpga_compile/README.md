# `Compiling SYCL* for FPGAs` Sample
This FPGA tutorial introduces how to compile SYCL*-compliant code for FPGAs through a simple vector addition example. If you are new to SYCL* for FPGAs, start with this sample.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Agilex®, Arria® 10, and Stratix® 10 FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | How and why compiling SYCL* code for FPGA differs from CPU or GPU <br> The compile options used to target FPGA devices
| Time to complete                  | 60 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For the simulation flow, one of the following simulators must be installed and accessible through your PATH environment variable:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
> 
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
> 
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

> **Note**: SYCL USM allocations, used in `part2` and `part3` of this tutorial, are only supported on FPGA boards that have a USM capable BSP (e.g. the Intel® FPGA PAC D5005 with Intel Stratix® 10 SX with USM support: intel_s10sx_pac:pac_s10_usm) or when targeting an FPGA family/part number.

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

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.



> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

## Purpose
Field-programmable gate arrays (FPGAs) are configurable integrated circuits that can be programmed to implement arbitrary circuit topologies. Classified as *spatial* compute architectures, FPGAs differ significantly from fixed Instruction Set Architecture (ISA) devices like CPUs and GPUs. FPGAs offer a different set of optimization trade-offs from these traditional accelerator devices.

While SYCL* code can be compiled for CPU, GPU, or FPGA, compiling to FPGA is somewhat different. This tutorial explains these differences and shows how to compile a "Hello World" style vector addition kernel for FPGA, following the recommended workflow.

### Why is compilation different for FPGA?
FPGAs differ from CPUs and GPUs in many interesting ways. 

Compared to CPU or GPU, generating a device image for FPGA hardware is a computationally intensive and time-consuming process. It is usual for an FPGA compile to take several hours to complete. For this reason, only ahead-of-time (or "offline") kernel compilation mode is supported for FPGA. The long compile time for FPGA hardware makes just-in-time (or "online") compilation impractical.

Long compile times are detrimental to developer productivity. The Intel® oneAPI DPC++/C++ Compiler provides several mechanisms that enable developers targeting FPGAs to iterate quickly on their designs. By circumventing the time-consuming process of full FPGA compilation wherever possible, developers can enjoy the fast compile times familiar to CPU and GPU developers.

### Multiarchitecture binary vs IP component

In the FPGA multiarchitecture binary generation flow, you can generate an executable host application and accelerator for a PCIe FPGA board if you have a compatible board support package (BSP). Intel provides BSPs for the Intel® PAC with Intel Arria® 10 GX FPGA, and the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX). If you have a different board, check with your vendor to see if they supply a BSP.

In the FPGA IP component generation flow, you can generate an IP component that you can import into an Intel® Quartus® Prime project. You can generate an IP by targeting your compilation to a supported Intel® FPGA device family or part number (for example, `Agilex` or `AGFA014R24B1E1V`) instead of a named board (for example, `intel_a10gx_pac:pac_a10`). 

The FPGA IP component generation flow does not generate any FPGA accelerated executable, only RTL (Register Transfer Level) IP component files. The host application is treated only as a 'testbench' that exercises and validates your IP component in emulation and simulation.

### Four compilation options
The four types of FPGA compilation are summarized in the table below.

   | Target          | Expected Time  | Output                                                                       | Description
   |:---             |:---            |:---                                                                          |:---
   |Emulator  | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | Optimization Report | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.
   | Simulator | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | FPGA Hardware | Multiple Hours | Quartus Place & Route (Multiarchitecture binary) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime. If you specified a BSP with `FPGA_DEVICE`, this will generate an FPGA image that you can run on the corresponding accelerator board.

The typical FPGA development workflow is to iterate in each of these stages, refining the code using the feedback provided by that stage. You can avoid long compile times by relying on emulation and the optimization report whenever possible.

#### FPGA Emulator

The FPGA emulator is the fastest method to verify the correctness of your code. The FPGA emulator executes the SYCL* device code on the CPU. The emulator is similar to the SYCL* host device, but unlike the host device, the FPGA emulator device supports FPGA extensions such as FPGA pipes and `fpga_reg` (although some of these features, such as `fpga_reg` may not affect how your design runs on the emulator).

There are two important caveats to remember when using the FPGA emulator.
* **Performance is not representative.** _Never_ draw inferences about FPGA performance from the FPGA emulator. The FPGA emulator's timing behavior is uncorrelated to that of the physical FPGA hardware. For example, an optimization that yields a 100x performance improvement on the FPGA may show no impact on the emulator performance. It may show an unrelated increase or even a decrease.
* **Undefined behavior may differ.** If your code produces different results when compiled for the FPGA emulator versus FPGA hardware, your code may exercises undefined behavior. By definition, undefined behavior is not specified by the language specification and may manifest differently on different targets.

#### Optimization Report (Early Image)

For this compilation type, your SYCL device code is optimized and converted into an FPGA design specified in Verilog RTL (a low-level, native entry language for FPGAs). This intermediate compilation result is also called the *FPGA early device image*, which is **not** executable. 

The optimization report contains significant information about how the compiler has transformed your device code into an FPGA design. The report includes visualizations of structures generated on the FPGA, performance and expected performance bottleneck information, and estimated resource utilization. Optimization reports are generated for the "optimization report", "simulator" and "hardware" compilation types. 

The [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/analyze-your-design.html) contains a chapter about how to analyze the reports generated after the FPGA early image and FPGA image.

#### FPGA Simulator
The FPGA simulator allows you to simulate the exact behavior of the synthesized kernel. Like emulation, you can run simulation on a system that does not have a target FPGA board installed. The simulator models a kernel much more accurately than the emulator, but it is much slower than the emulator.

The Intel oneAPI DPC++/C++ Compiler links your design C++ testbench with an RTL-compiled version of your component that runs in an RTL simulator. You do not need to invoke any RTL simulator manually, but you can add the `-Xsghdl` flag to save the simulation waveform for later viewing.

> **Note**: Running the simulation executable can take a long time if your device code is complex or if your test inputs are large. To save simulation time, use the smallest possible input.

#### FPGA Hardware (Hardware Image)

The generated Verilog RTL is mapped onto the FPGA hardware resources by the Intel® Quartus® Prime software. The estimated performance and resource utilization is therefore much more accurate than the estimates obtained in the optimization report compilation type. 

If you compile a multiarchitecture binary, the resulting binary will include an FPGA hardware image (also referred to as a bitstream) that is executable on an FPGA accelerator card with a supported BSP. The compiler will interface your design with the BSP, and your host application will seamlessly make the system calls to launch kernels on the FPGA.

If you compile an IP component, the compilation result is **not** executable. IP components are compiled in isolation and not interfaced with other components on the FPGA. The purpose of this compilation flow is to get accurate resource utilization and performance data for IP components.

This compilation process takes hours, although it may be faster if you generate a re-usable IP component.

### Device Selectors
The following code snippet demonstrates how you can specify the target device in your source code. The selector is used to specify the target device at runtime.

It is recommended to use a preprocessor macro to choose between the emulator and FPGA selectors. This makes it easy to switch between targets using only command-line options. Since the FPGA only supports ahead-of-time compilation, dynamic selectors (such as the `default_selector`) are less useful than explicit selectors when targeting FPGA.

```c++
// FPGA device selectors are defined in this utility header
#include <sycl/ext/intel/fpga_extensions.hpp>

int main() {
// choose a selector based on compiler flags.
#if FPGA_SIMULATOR
        auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
        auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
        sycl::queue q(selector);
    ...
}
```

### Compiler Options
This section includes a helpful list of commands and options to compile this design for the FPGA emulator, generate the FPGA early image optimization reports, and compile for FPGA hardware.
>**Note**: In this sample, the compiler is refered to as `icpx`. On Windows, you should use `icx-cl`.

FPGA Emulator 

```bash
# FPGA emulator image
icpx -fsycl -fintelfpga -DFPGA_EMULATOR -I../../../../include vector_add.cpp -o vector_add.fpga_emu
```

Optimization Report

```bash
# FPGA early image (with optimization report):
icpx -fsycl -fintelfpga -DFPGA_HARDWARE -I../../../../include vector_add.cpp -Xshardware -fsycl-link=early -Xstarget=Agilex -o vector_add_report.a
```
Use the`-Xstarget` flag to target a supported board, a device family, or a specific FPGA part number.

Simulator

```bash
# FPGA simulator image:
icpx -fsycl -fintelfpga -DFPGA_SIMULATOR -I../../../../include vector_add.cpp -Xssimulation -Xstarget=Agilex -Xsghdl -o vector_add_sim.a
```
Through `-Xstarget`, you can target an explicit board, a device family or a FPGA part number.

Hardware

```bash
# FPGA hardware image:
icpx -fsycl -fintelfpga -DFPGA_HARDWARE -I../../../../include vector_add.cpp -Xshardware -Xstarget=Agilex -o vector_add.fpga
```
Through `-Xstarget`, you can target an explicit board, a device family or a FPGA part number.

`-DFPGA_EMULATOR`, `-DFPGA_SIMULATOR`, `-DFPGA_HARDWARE` are options that adds a preprocessor define that invokes the emulator/simulator/FPGA device selector in this sample (see code snippet above).

The [Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/programming-interface/fpga-flow/fpga-compilation-flags.html) contains a chapter explains the compiler options used here.

### Source Code

There are 4 parts to this tutorial located in the 3 sub-folders. Together, they demonstrate how you can migrate an algorithm from vanilla C++ code to SYCL for FPGA. Note that you may also choose to use a functor with buffers, or a function with USM.

#### Part 1 C++
Part 1 demonstrates a vector addition program in vanilla C++. Observe how the `VectorAdd` function is separated from the `main()` function, and the `vec_a`, `vec_b`, and `vec_c` vectors are allocated onto the heap.

#### Part 2 SYCL* (functor and USM)
Part 2 shows the same vector addition from part 1, but in SYCL* C++ with a 'functor' coding style using a unified shared memory (USM) interface. Compare with the source code in part 1 to see SYCL*-specific code changes. Observe how the `VectorAdd` functor is called using `q.single_task<...>(VectorAdd{...});`. This tells the DPC++ compiler to convert `VectorAdd` into RTL. Also observe how `vec_a`, `vec_b`, and `vec_c` are allocated into a shared memory space using `malloc_shared`; this tells the DPC++ compiler that `vec_a`, `vec_b`, and `vec_c` should be visible both to your kernel and your host code (or if you are creating an IP component, the testbench code).

#### Part 3 SYCL* (lambda function and USM)
Part 3 demonstrates vector addition in SYCL* C++ with a 'function' coding style using unified shared memory (USM). This code style will be familiar to users who are already experienced with SYCL*. Observe how the `VectorAdd` function is called using a lambda expression:
```c++
h.single_task<...>([=]() {
    VectorAdd(...);
});
```
#### Part 4 SYCL* (lambda function and buffer)
Part 4 shows the vector addition in SYCL* C++ with a 'function' coding style and buffer & accessor interface. This code style will be familiar to users who are already experienced with SYCL*. Observe how `vec_a`, `vec_b`, and `vec_c` are copied into buffers before the `VectorAdd` function is called.

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

### Run CMake to generate the Makefiles

#### On a Linux* System
For different parts of this tutorial, navigate to the appropriate sub-folder.
```bash
cd <partX_XXX>
```
`<partX_XXX>` can be:
- `part1_cpp`
- `part2_dpcpp_functor_usm`
- `part3_dpcpp_lambda_usm`
- `part4_dpcpp_lambda_buffers`

Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex® device family), run `cmake` using the command:
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

#### On a Windows* System
For different parts of this tutorial, navigate to the appropriate sub-folder.
```cmd
cd <partX_XXX>
```
`<partX_XXX>` can be:
- `part1_cpp`
- `part2_dpcpp_functor_usm`
- `part3_dpcpp_lambda_usm`
- `part4_dpcpp_lambda_buffers`

Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex® device family), run `cmake` using the command:
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

### Build using `make`/`nmake`
After using CMake to generate build artifacts, you can then build with specific targets. This project can build 4 targets.

| Compilation Type    | Command (Linux)    | Command (Windows)
|:---                 |:---                |:---
| FPGA Emulator       | `make fpga_emu`    | `nmake fpga_emu`
| Optimization Report | `make report`      | `nmake report`
| FPGA Simulator      | `make fpga_sim`    | `nmake fpga_sim`
| FPGA Hardware       | `make fpga`        | `nmake fpga`

The `fpga_emu`, `fpga_sim` and `fpga` targets produce binaries that you can run. The executables will be called `vector_add.fpga_emu`, `vector_add.fpga_sim`, and `vector_add.fpga`. The `fpga` target will produce an executable binary if you create a multiarchitecture binary kernel.

For part 1 of this tutorial, only the `fpga_emu` target is available as this regular C++ code only  target a CPU.

## Examining the Reports
In *part2*, *part3* and *part4*, after running the `report` target, the optimization report can be viewed using the `fpga_report` application:
```
fpga_report vector_add.report.prj/reports/vector_add_report.zip
```
Browse the reports that were generated for the `VectorAdd` kernel's FPGA early image. You may also wish to examine the reports generated by the simulation compile and full FPGA hardware compile and compare their contents.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./vector_add.fpga_emu     (Linux)
     vector_add.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA simulator device (the kernel executes in a simulator):
   * On Linux
      ```
      CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./vector_add.fpga_sim
      ```
   * On Windows
      ```
      set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
      vector_add.fpga_sim.exe
      set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
      ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`):
     ```
     ./vector_add.fpga         (Linux)
     vector_add.fpga.exe       (Windows)
     ```

### Example of Output
```
using FPGA Simulator.
add two vectors of size 256
PASSED
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
