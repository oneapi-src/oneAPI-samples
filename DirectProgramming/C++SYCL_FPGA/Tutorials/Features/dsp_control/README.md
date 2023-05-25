# `DSP Control` Sample

This sample is an FPGA tutorial that demonstrates how to set the implementation preference for certain math operations (addition, subtraction, and multiplication) between hardened DSP blocks and soft logic.

| Area                  | Description
|:---                   |:---
| What you will learn   |  How to apply global DSP control in command-line interface. <br> How to apply local DSP control in source code. <br> Scope of datatypes and math operations that support DSP control.
| Time to complete      | 15 minutes
| Category              | Concepts and Functionality

## Purpose

This tutorial shows how to apply global and local controls to set the implementation preference between DSPs and soft-logic for certain math operations. The global control is applied using a command-line flag and affects applicable math operations in all kernels. The local control is applied as a library function and affects math operations in a block scope in a single kernel. Both global and local controls only affect math operations that support DSP control (see table below).

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

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.


## Key Implementation Details

The sample illustrates these important concepts.

- Scope of data types and math operations that support DSP control.
- How to apply global DSP control from the command-line.
- How to apply local DSP control in source code.

### Scope of Data Types and Math Operations

| Datatype           | Controllable Math Operations
|:---                |:---
| `float`            | addition, subtraction, multiplication
| `ap_float<8, 23>`  | addition, subtraction, multiplication
| `int`              | multiplication
| `ac_int`           | multiplication
| `ac_fixed`         | multiplication

### Global Control

The `-Xsdsp-mode=<option>` command-line flag sets implementation preference of math operations that support DSP control in all kernels. It has three valid options:

| Option             | Explanation
|:---                |:---
| `default`          | This is the default option if this command-line flag is not passed manually. The compiler determines the implementation based on datatype and math operation.
| `prefer-dsp`       | Prefer math operations to be implemented in **DSPs**. If a math operation is implemented by DSPs by default, you will see no difference in resource utilization or area. Otherwise, you will notice a decrease in the usage of soft-logic resources and an increase in the usage of DSPs.
| `prefer-softlogic` | Prefer math operations to be implemented in **soft-logic**. If a math operation is implemented without DSPs by default, you will see no difference in resource utilization or area. Otherwise, you will notice a decrease in the usage of DSPs and an increase in the usage of soft-logic resources.

### Local Control

The library function `math_dsp_control<Preference::<enum>, Propagate::<enum>>([&]{})` provides block scope local control in one single kernel. A reference-capturing lambda expression is passed as the argument to this library function. Inside the lambda expression, implementation preference of math operations that support DSP control will be determined by two template arguments.

The first template argument `Preference::<option>` is an enum with three valid options:

| Option               | Explanation
|:---                  |:---
| `DSP`                | Prefer math operations to be implemented in **DSPs**. Its behavior on a math operation is equivalent to global control `-Xsdsp-mode=prefer-dsp`. <br> **NOTE:** This option will be automatically applied if the template argument `Preference` is not specified manually.
| `Softlogic`          | Prefer math operations to be implemented in **soft-logic**. Its behavior on a math operation is equivalent to global control `-Xsdsp-mode=prefer-softlogic`.
| `Compiler_default`   | Compiler determines the implementation based on datatype and math operation. Its behavior on a math operation is equivalent to global control `-Xsdsp-mode=default`.

The second template argument `Propagate::<option>` is an enum that determines whether the DSP control applies to controllable math operations in function calls inside the lambda expression:

| Option               | Explanation
|:---                  |:---
| `On`                 | DSP control recursively applies to controllable math operations in all function calls inside the lambda expression. <br> **NOTE:** This option will be automatically applied if the template argument `Propagate` is not specified manually.
| `Off`                | DSP control only applies to controllable math operations directly inside the lambda expression. Math operations in function calls inside the lambda expression are not affected by this DSP control.

>**Note**: A nested `math_dsp_control<>()` call is only controlled by its own `Preference`. The `Preference` of the parent `math_dsp_control<>()` does not affect the nested `math_dsp_control<>()`, even if the parent has `Propagate::On`. Additionally, local control overrides global control on a controlled math operation.

## Build the `DSP Control` Tutorial

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

Locate `report.html` in the `dsp_control_report.prj/reports/` directory. 

1. Navigate to **Area Analysis of System** (Area Analysis > Area Analysis of System). In this view, you can see usage of FPGA resources, which reflects the outcome of DSP control.
2. Navigate to **System Viewer** (Views > System Viewer). In this view, you can verify the `Implementation Preference` on the Details Panel of the graph node of a controlled math operation.

## Run the `DSP Control` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./dsp_control.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./dsp_control.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./dsp_control.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   dsp_control.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   dsp_control.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   dsp_control.fpga.exe
   ```

## Example Output

```
PASSED: all kernel results are correct.
```

Experiment with the code in this tutorial.

- Try various control options with global control and local control.
- Try various data types and math operations that support DSP control.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).