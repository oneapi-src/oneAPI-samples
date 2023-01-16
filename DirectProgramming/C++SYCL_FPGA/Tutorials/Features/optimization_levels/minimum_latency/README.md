# Minimum Latency Flow

This FPGA tutorial demonstrates how to compile your design with the minimum latency flow to achieve low latency at the cost of increased f<sub>MAX</sub>.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg; 10 GX FPGA <br> Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX) <br> Intel&reg; FPGA 3rd party / custom platforms with oneAPI support <br> **Note**: Intel&reg; FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | How to use the minimum latency flow to compile low-latency designs<br>How to manually override underlying controls set by the minimum latency flow
| Time to complete                  | 20 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

## Purpose

This FPGA tutorial demonstrates how to use the minimum latency flow to compile low-latency designs and how to manually override underlying controls set by the minimum latency flow. By default, the minimum latency flow tries to achieve lower latency at the cost of decresed f<sub>MAX</sub>, so this flow is a good starting point for optimizing latency-sensitive designs.

To compile your design with the minimum latency flow, pass the `-Xsoptimize=latency` flag to the `icpx` command.

The minimum latency flow implies the following compiler controls:
- Disable hyper-optimized handshaking on Intel Stratix&reg; 10 and Intel Agilex&trade; devices
- Use zero-latency stall-free clusters exit FIFO
- Disable loop speculation
- Remove the 1-cycle delay on the pipelined loop limiter

The following table shows how users can manually override these underlying controls:
||Control Flags/Attributes|Reference
|:---|:---|:---
|Hyper-optimized handshaking|`-Xshyper-optimized-handshaking=<auto\|off\|on>`|[Modify the Handshaking Protocol Between Clusters](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/optimization-flags/hyper-opt-handshaking.html)
|Exit FIFO latency of stall-free clusters|`-Xssfc-exit-fifo-type=<default\|zero-latency\|low-latency>`|[Global Control of Exit FIFO Latency of Stall-free Clusters](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/optimization-flags/control-exit-fifo-latency.html)
|Loop speculation|`[[intel::speculated_iterations(N)]]`|[`speculated_iterations` Attribute](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/loop-directives/speculated-iterations-attribute.html) 
|Pipelined loop limiter|N/A|N/A

> **Note**: The more specific manual control overrides minimum latency flow's default control.

### Understanding the Tutorial Design

The basic function performed by the tutorial kernel is a RGB to grayscale algorithm. To see the impact of the minimum latency flow in this tutorial in terms of latency and f<sub>MAX</sub>, and also see how to override the minimum latency flow with specific manual controls, the design needs to be compiled three times.

Part 1 compiles the design without passing the `-Xsoptimize=latency` flag. In this default flow, the compiler targets higher throughput and f<sub>MAX</sub> with the sacrifice of latency and area.

Part 2 compiles the design with the `-Xsoptimize=latency` flag, so the minimum latency flow is used in this compile. By setting up the underlying compiler controls listed above, the minimum latency flow achieves lower latency by trading off f<sub>MAX</sub>.

Part 3 also compiles the design with the minimum latency flow, as well as manual controls that revert minimum latency flow's default underlying controls. Therefore, latency and f<sub>MAX</sub> of this compile are the same as part 1.

### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.

## Key Concepts

* How to use the minimum latency flow to compile low-latency designs
* How to manually override underlying controls set by the minimum latency flow

## Building the `minimum_latency` Tutorial

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see **Use the setvars Script** for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel&reg; oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel&reg; oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On a Linux* System

1. Generate the `Makefile` by running `cmake`:

   ```bash
   mkdir build
   cd build
   ```

   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:

   ```bash
   cmake ..
   ```

   Alternatively, to compile for the Intel&reg; PAC D5005 (with Intel Stratix&reg; 10 SX FPGA), run `cmake` using the command:

   ```bash
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```bash
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

2. Compile the design using the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):

     ```bash
     make fpga_emu
     ```

   * Generate the optimization reports:

     ```bash
     make report
     ```

   * Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):

     ```bash
     make fpga_sim
     ```

   * Compile for FPGA hardware (longer compile time, targets FPGA device):

     ```bash
     make fpga
     ```

3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/minimum_latency.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
   ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel&reg; PAC D5005 (with Intel Stratix&reg; 10 SX FPGA), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   * Generate the optimization reports:
     ```
     nmake report
     ```
   * Compile for simulation (fast compile time, targets simulated FPGA device, reduced data  size):
     ```
     nmake fpga_sim
     ``
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

> **Note**: The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).

## Examining the Reports

Locate the pair of `report.html` files in either:

* **Report-only compile**:  `no_control_report.prj`, `minimum_latency_report.prj`, and `manual_revert_report.prj`
* **FPGA hardware compile**: `no_control.fpga.prj`, `minimum_latency.fpga.prj`, and `manual_revert.fpga.prj`

Open the reports in Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to **Loop Analysis** (**Throughput Analysis > Loop Analysis**). In this viewer, you can find the latency of loops in the kernel. The latency of the compile with the minimum latency flow (part 2) should be smaller than the other two compiles. Also, the latency of the other two compiles (part 1 & 3) should be the same.

Navigate to **Clock Frequency Summary** (**Summary > Clock Frequency Summary**) in `no_control.fpga.prj/reports/report.html`, `minimum_latency.fpga.prj/reports/report.html`, and `manual_revert.fpga.prj/reports/report.html` (after `make fpga` completes). In this table, you can find the actual f<sub>MAX</sub>. The f<sub>MAX</sub> of the compile with the minimum latency flow (part 2) should be smaller than the other two compiles. Also, the f<sub>MAX</sub> of the other two compiles (part 1 & 3) should be the same. Note that only the report generated by the FPGA hardware compile will reflect the true f<sub>MAX</sub> affected by the minimum latency flow. The difference is **not** apparent in the reports generated by `make report` because a design's f<sub>MAX</sub> cannot be predicted.

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):

   ```bash
   ./no_control.fpga_emu               (Linux)
   no_control.fpga_emu.exe             (Windows)
   ```

2. Run the sample on the FPGA simulator device:
   ```bash
   # Sample without minimum latency flow
   ./no_control.fpga_sim               (Linux)
   no_control.fpga_sim.exe             (Windows)
   # Sample with minimum latency flow
   ./minimum_latency.fpga_sim          (Linux)
   minimum_latency.fpga_sim.exe        (Windows)
   # Sample with minimum latency flow but controls manually reverted
   ./manual_revert.fpga_sim            (Linux)
   manual_revert.fpga_sim.exe          (Windows)

3. Run the sample on the FPGA device

   ```bash
   # Sample without minimum latency flow
   ./no_control.fpga                   (Linux)
   no_control.fpga.exe                 (Windows)
   # Sample with minimum latency flow
   ./minimum_latency.fpga              (Linux)
   minimum_latency.fpga.exe            (Windows)
   # Sample with minimum latency flow but controls manually reverted
   ./manual_revert.fpga                (Linux)
   manual_revert.fpga.exe              (Windows)
   ```

### Example of Output

Output of sample without minimum latency flow:
```txt
Kernel Throughput: 195.716MB/s
Exec Time: 1.9491e-05s, InputMB: 0.0038147MB
PASSED: all kernel results are correct
```

Output of sample with minimum latency flow:
```txt
Kernel Throughput: 137.764MB/s
Exec Time: 2.769e-05s, InputMB: 0.0038147MB
PASSED: all kernel results are correct
```

Output of sample with minimum latency flow but controls manually reverted:
```txt
Kernel Throughput: 192.934MB/s
Exec Time: 1.9772e-05s, InputMB: 0.0038147MB
PASSED: all kernel results are correct
```

### Discussion of Results

Comparing to Intel Arria&reg; 10 GX FPGA, it is more notable on Intel Stratix&reg; 10 SX FPGA that the minimum latency flow significantly reduces the latency, along with the f<sub>MAX</sub> and the throughput. That is because the minimum latency flow disables the hyper-optimized handshaking, which achieves higher f<sub>MAX</sub> at the cost of increased latency. For more information on the hyper-optimized handshaking protocol on Intel Stratix&reg; 10 and Intel Agilex&trade; devices, see [Modify the Handshaking Protocol Between Clusters](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/flags-attr-prag-ext/optimization-flags/hyper-opt-handshaking.html).

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
