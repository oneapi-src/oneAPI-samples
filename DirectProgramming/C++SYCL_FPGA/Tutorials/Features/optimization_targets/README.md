# `Optimization Targets` Sample

This sample is an FPGA tutorial that demonstrates how to set optimization targets for your compile to target different performance metrics. 

This tutorial shows compiling with the minimum latency optimization target to achieve low latency at the cost of reduced f<sub>MAX</sub>.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to set optimization targets for your compile. </br> How to use the minimum latency optimization target to compile low-latency designs. </br> How to manually override underlying controls set by the minimum latency optimization target.
| Time to complete       | 20 minutes
| Category               | Concepts and Functionality

## Purpose

This FPGA tutorial demonstrates how to set optimization targets for your compile to target different performance metrics.

The `-Xsoptimize=<flag>` command-line option sets optimization targets, and it supports the following flags:

|Flag                      |Explanation                        |Documentation
|:---                      |:---                               |:---
|`latency`                 |Minimum latency: minimize kernel latency at the cost of decreased f<sub>MAX</sub> |[*Minimum Latency Flow*](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/developer-guide/current/minimum-latency-flow.html)
|`throughput-area-balanced`|Balanced throughput-area trade-offs: disable throughput-area trade-off heuristics that increase the throughput at the cost of area |[*Balanced Throughput-Area Trade-Offs Flow*](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/developer-guide/current/max-throughput.html)
|`area`                    |Minimum area: minimize kernel area at the cost of decreased f<sub>MAX</sub> |[*Minimum Area Flow*](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/developer-guide/current/minimum-area.html)

To compile your design with the minimum latency optimization target, use the flag option `-Xsoptimize=latency`.

As an example, this tutorial shows how to use the minimum latency optimization target to compile low-latency designs and how to manually override underlying controls set by the minimum latency optimization target. By default, the minimum latency optimization target tries to achieve lower latency at the cost of decreased f<sub>MAX</sub>, so it is a good starting point for optimizing latency-sensitive designs.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware             | Intel® Agilex® 7, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.

> **Warning**: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

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

The sample illustrates the following important concepts.

- Setting optimization targets to use when compiling your program.
- Using the minimum latency optimization target to compile low-latency designs.
- Manually overriding underlying controls set by the minimum latency optimization target.

### Understanding the Tutorial Design

The basic function performed by the tutorial kernel is an RGB to grayscale algorithm. We compile the design three times to see the impact of the minimum latency optimization target in this tutorial in terms of latency and f<sub>MAX</sub> and to see how to override underlying controls set by the minimum latency optimization target with specific manual controls.

- Part 1 compiles the design without the `-Xsoptimize=latency` flag. In this default flow, the compiler targets higher throughput and f<sub>MAX</sub> with the sacrifice of latency and area.

- Part 2 compiles the design with the `-Xsoptimize=latency` flag, so the minimum latency optimization target is used in this compile, which lowers latency by trading off f<sub>MAX</sub>.

- Part 3 compiles the design with the minimum latency optimization target and includes manual controls that revert default underlying controls set by the minimum latency optimization target. Therefore, latency and f<sub>MAX</sub> of this compile are the same as part 1.

## Build the `Optimization Targets` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

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
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your 'build' directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory, for example:
>
>  ```
  > C:\samples\build> cmake -G "NMake Makefiles" C:\long\path\to\code\sample\CMakeLists.txt
>  ```
### Read the Reports

Locate the `report.html` files in the following locations (depending on the compile path that you selected):

- **Report-only compile**: `no_control_report.prj`, `minimum_latency_report.prj`, and `manual_revert_report.prj`
- **FPGA hardware compile**: `no_control.fpga.prj`, `minimum_latency.fpga.prj`, and `manual_revert.fpga.prj`

Navigate to **Loop Analysis** (**Throughput Analysis > Loop Analysis**). In this viewer, you can find the latency of loops in the kernel. The latency of the compile with the minimum latency optimization target (part 2) should be lower than the other two compiles. Also, the latency of the other two compiles (part 1 & 3) should be the same.

Navigate to **Clock Frequency Summary** (**Summary > Clock Frequency Summary**) in `no_control.fpga.prj/reports/report.html`, `minimum_latency.fpga.prj/reports/report.html`, and `manual_revert.fpga.prj/reports/report.html` (after `make fpga` completes). In this table, you can find the actual f<sub>MAX</sub>. The f<sub>MAX</sub> of the compile with the minimum latency optimization target (part 2) should be lower than the other two compiles. Also, the f<sub>MAX</sub> of the other two compiles (part 1 & 3) should be the same. Note that only the report generated by the FPGA hardware compile will reflect the true f<sub>MAX</sub> affected by the minimum latency optimization target. The difference is **not** apparent in the reports generated by `make report` because a design's f<sub>MAX</sub> cannot be predicted.

## Run the `Optimization Targets` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./no_control.fpga_emu
   ```

2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./no_control.fpga_sim
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./minimum_latency.fpga_sim
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./manual_revert.fpga_sim
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./no_control.fpga
   ./minimum_latency.fpga
   ./manual_revert.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   no_control.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   no_control.fpga_sim.exe
   minimum_latency.fpga_sim.exe
   manual_revert.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   no_control.fpga.exe
   minimum_latency.fpga.exe
   manual_revert.fpga.exe
   ```

## Example Output

Example output without minimum latency optimization target:

```
Kernel Throughput: 195.716MB/s
Exec Time: 1.9491e-05s, InputMB: 0.0038147MB
PASSED: all kernel results are correct
```

Example output with minimum latency optimization target:

```
Kernel Throughput: 137.764MB/s
Exec Time: 2.769e-05s, InputMB: 0.0038147MB
PASSED: all kernel results are correct
```

Example output with minimum latency optimization target but controls manually reverted:

```
Kernel Throughput: 192.934MB/s
Exec Time: 1.9772e-05s, InputMB: 0.0038147MB
PASSED: all kernel results are correct
```

Comparing to Intel® Arria® 10 GX FPGA, it is more notable on Intel® Stratix® 10 SX FPGA that the minimum latency optimization target significantly reduces the latency, along with the f<sub>MAX</sub> and the throughput. That is because the minimum latency optimization target disables the hyper-optimized handshaking, which achieves higher f<sub>MAX</sub> at the cost of increased latency. 

> **Note**: For more information on the hyper-optimized handshaking protocol on Intel® Stratix® 10 and Intel Agilex® 7 devices, see the [*Modify the Handshaking Protocol Between Clusters (-Xshyper-optimized-handshaking)*](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/optimization-guide/current/hyper-opt-handshaking.html) topic in the *FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide*.

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).