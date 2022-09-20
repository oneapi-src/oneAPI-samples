# Latency Control
This FPGA tutorial demonstrates how to set latency constraints to pipes and load-store units (LSUs) accesses.

| Optimized for                     | Description
|:---                                 |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg;10 GX FPGA <br> Intel&reg;FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg;10 SX) <br> Intel&reg;FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel&reg;FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel&reg;oneAPI DPC++ Compiler <br> Intel&reg;FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | How to set latency constraints to pipes and LSUs accesses.<br>How to confirm that the compiler respected the latency control directive.
| Time to complete                  | 15 minutes

## Purpose
This FPGA tutorial demonstrates how to set latency constraints to pipes and LSUs accesses and how to confirm that the compiler respected the latency control directive.

### Use Model
> **Note**: The APIs described in this section are experimental. Future versions of latency controls might change these APIs in ways that are incompatible with the version described here.

Latency controls APIs are provided on member functions `read()` and `write()` of class `sycl::ext::intel::experimental::pipe` and on member functions `load()` and `store()` of class `sycl::ext::intel::experimental::lsu`. Other than the latency controls support, the experimental `pipe` and `lsu` are identical to `sycl::ext::intel::pipe` and `sycl::ext::intel::lsu`. The experimental `pipe` and `lsu` are also provided by `<sycl/ext/intel/fpga_extensions.hpp>`.

These `read()`, `write()`, `load()`, and `store()` member functions can take a property list instance (`sycl::ext::oneapi::experimental::properties`) as a function argument, which can contain the following latency controls properties:

* `sycl::ext::intel::experimental::latency_anchor_id<N>`, where `N` is a signed integer

   A label that can be associated with the pipes and LSUs functions listed above. This label can then be referenced by the `latency_constraint` properties to define relative latency constraints. Functions with this property will be referred to as "labeled functions".

* `sycl::ext::intel::experimental::latency_constraint<A, B, C>`

    A constraint then can be associated with the pipes and LSUs functions listed above. It provides a latency constraint between this function and a different labeled function. Functions which have this property will be referred to as "constrained functions". This constraint has  the following parameters:

   * `A` is a signed integer: The label of the labeled function which is constrained relative to the constrained function.
   * `B` is an enum value: The type of constraint, can be `latency_control_type::exact` (exact latency), `latency_control_type::max` (maximum latency), and `latency_control_type::min` (minimum latency).
   * `C` is a signed integer: The relative clock cycle difference between the labeled function and the constrained function, that the constraint should infer subject to the type of constraint (exact/max/main).

### Simple Code Example
The following example shows you how to use latency controls on pipes. The example also uses a function acting as both labeled function and constrained function:
```cpp
using Pipe1 = sycl::ext::intel::experimental::pipe<class PipeClass1, int, 8>;
using Pipe2 = sycl::ext::intel::experimental::pipe<class PipeClass2, int, 8>;
using Pipe3 = sycl::ext::intel::experimental::pipe<class PipeClass2, int, 8>;
...
// In kernel:
// The following read has a label 0.
int value = Pipe1::read(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::latency_anchor_id<0>));

// The following write occurs exactly 2 cycles after the label-0 function, i.e.,
// the read above. Also, it has a label 1.
Pipe2::write(
    value,
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::latency_anchor_id<1>,
        sycl::ext::intel::experimental::latency_constraint<
            0, sycl::ext::intel::experimental::latency_control_type::exact,
            2>));

// The following write occurs at least 2 cycles after the label-1 function,
// i.e., the write above.
Pipe3::write(
    value,
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::latency_constraint<
            1, sycl::ext::intel::experimental::latency_control_type::min, 2>));
```

This next example shows you how to use latency controls on LSUs. It also uses a negative relative cycle number in `latency_constraint`, which means that the constrained function is scheduled **before** the associated labeled function:
```cpp
using BurstCoalescedLSU = sycl::ext::intel::experimental::lsu<
    sycl::ext::intel::experimental::burst_coalesce<false>,
    sycl::ext::intel::experimental::statically_coalesce<false>>;
...
// In kernel:
// The following load occurs at most 5 cycles before the label-2 function,
// i.e., the store below.
int value = BurstCoalescedLSU::load(
    input_ptr,
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::latency_constraint<
            2, sycl::ext::intel::experimental::latency_control_type::max, -5>));

// The following store has a label 2.
BurstCoalescedLSU::store(
    output_ptr, value,
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::latency_anchor_id<2>));
```

### Rules and Limitations
* `latency_anchor_id` must be non-negative
* `latency_anchor_id` must be a unique number within the whole design
* The labeled function and the constrained function of a constraint must meet one of the following conditions:
  * Both are in the same block but not in any cluster
  * Both are in the same cluster

> **Note**: Clusters can be identified in the System Viewer (**Views > System Viewer**) of the `report.html` report.

The compiler tries to achieve the latency constraints, and it errors out if some constraints cannot be satisfied. For example, if one constraint specifies function A should be scheduled after function B, while another constraint specifies function B should be scheduled after function A, then that set of constraints is unsatisfiable.

### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.

## Key Concepts
 * How to set latency constraints to pipes and LSUs accesses.
 * How to confirm that the compiler respected the latency control directive.

## Building the `latency_control` Tutorial
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

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel&reg;oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel&reg;oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg;PAC with Intel Arria&reg;10 GX FPGA, run `cmake` using the command:
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel&reg;FPGA PAC D5005 (with Intel Stratix&reg;10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Generate the optimization report:
     ```
     make report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/latency_control.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg;PAC with Intel Arria&reg;10 GX FPGA, run `cmake` using the command:
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel&reg;FPGA PAC D5005 (with Intel Stratix&reg;10 SX), run `cmake` using the command:

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

> **Note**: The Intel&reg;PAC with Intel Arria&reg;10 GX FPGA and Intel&reg;FPGA PAC D5005
(with Intel Stratix&reg;10 SX) do not support Windows*. Compiling to FPGA hardware
on Windows* requires a third-party or custom Board Support Package (BSP) with
Windows* support.

> **Note**: If you encounter any issues with long paths when
compiling under Windows*, you may have to create your ‘build’ directory in a
shorter path, for example c:\samples\build. You can then run cmake from that
directory, and provide cmake with the full path to your sample directory.

### Troubleshooting
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel&reg;oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel&reg;oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html)

## Examining the Reports
Locate `report.html` in the `latency_control_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

1. Navigate to Schedule Viewer (**Views > Schedule Viewer**). In this view, you can find the load and the store that have latency constraint and then you can check the scheduled latency between them. The scheduled latency should reflect the applied latency control in the design source code. You can also verify the latency control parameters in the **Details** pane for the load and store nodes.

2. Navigate to System Viewer (**Views > System Viewer**). In this view, you can verify the latency control parameters in the **Details** pane for the load and the store nodes. You can find the load and store nodes in **LatencyControl.B1**.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./latency_control.fpga_emu     (Linux)
     latency_control.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./latency_control.fpga         (Linux)
     latency_control.fpga.exe       (Windows)
     ```

### Example of Output
```
PASSED: all kernel results are correct.
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).