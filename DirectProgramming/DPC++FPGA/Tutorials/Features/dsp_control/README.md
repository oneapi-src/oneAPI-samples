# DSP Control
This FPGA tutorial demonstrates how to set implementation preference for math operations, which affects hardware resource utilization on FPGA device.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               |  How to apply global DSP control in command-line interface. <br> How to apply local DSP control in source code. <br> Scope of datatypes and math operations that support DSP control.
| Time to complete                  | 15 minutes

## Purpose
This tutorial shows how to apply global and local DSP controls to set implementation preference for math operations to control usage of DSPs. The global control is applied via command-line option and affects math operations in all kernels. The local control is applied via library function and affects math operations in a block scope in one single kernel. Both global and local controls only affect math operations that support DSP control.

### Scope of Datatypes and math operations
| Datatypes              | math operations
---                      |---
| `float`                | Add, sub, constant mul
| `hls_float<8, 23>`     | Add, sub, constant mul
| `int`                  | Constant mul
| `ac_int`               | Constant mul
| `ac_fixed`             | Constant mul

**NOTE:** _constant mul_ means one operand of the multiplication is a constant.

### Global Control
The `-Xsdsp-mode` command-line option sets implementation preference of math operations that support DSP control in all kernels. It has three valid options:
| Option                           | Explanation
---                                |---
| `-Xsdsp-mode=default`            | This is the default. The compiler determines the implementation based on datatype and math operation.
| `-Xsdsp-mode=prefer-dsp`         | Prefer math operations to be implemented in **DSPs**. If a math operation is implemented by DSPs by default, you will see no difference in resource utilization or area. Otherwise, you will notice decrease in the usage of soft-logic resources and increase in the usage of DSPs.
| `-Xsdsp-mode=prefer-softlogic`   | Prefer math operations to be implemented in **soft-logic**. If a math operation is implemented without DSPs by default, you will see no difference in resource utilization or area. Otherwise, you will notice decrease in the usage of DSPs and increase in the suage of soft-logic resources.

### Local Control
The library function `math_dsp_control<Preference::<enum>, Propagate::<bool>>([&]{})` provides block scope local control in one single kernel. A reference-capturing lambda expression is passed as the argument to this library function. Inside the lambda expression, implementation preference of math operations that support DSP control will be determined by two template arguments.

The first template argument `Preference` is an enum with three valid options:
| Option                           | Explanation
---                                |---
| `Preference::DSP`                | This is the default. Prefer math operations to be implemented in **DSPs**. Its behavior on a math operation is equivalent to global control `-Xsdsp-mode=prefer-dsp`.
| `Preference::Softlogic`          | Prefer math operations to be implemented in **soft-logic**. Its behavior on a math operation is equivalent to global control `-Xsdsp-mode=prefer-softlogic`.
| `Preference::Compiler_default`   | Compiler determines the implementation based on datatype and math operation. Its behavir on a math operation is equivalent to global control `-Xsdsp-mode=default`.

The second template argument `Propagate` is a boolean that determines propagation of the first argument `Preference` to functions called in the lambda expression:
| Option  | Explanation
---       |---
| `On`    | This is the default. `Preference` is propagated to all functions called in the lambda expression, and affects all math operations on the call graph in the lambda expression until a nested `math_dsp_control<>()` call stops the propatation.
| `Off`   | `Preference` is not propagated to functions called in the lambda expression. So `Preference` only applies to math operations directly used in the lambda expression.

**NOTE:**
1. `Preference` never applies to nested `math_dsp_control<>()` calls. Each nested `math_dsp_control<>()` has its own `Preference`.
2. Local control overrides global control on a controlled math operation.

## Key Concepts
* How to apply global DSP control in command-line interface
* How to apply local DSP control in source code
* Scope of datatypes and math operations that support DSP control

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `dsp_control` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

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
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/dsp_control.fpga.tar.gz" download>here</a>.

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

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.
 
 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `dsp_control_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

1. Navigate to Area Analysis of System (Area Analysis > Area Analysis of System). In this view, you can see usage of FPGA resources which reflects the outcome of DSP control.
2. Navigate to System Viewer (Views > System Viewer). In this view, you can verify the `Implementation Preference` on the Details Panel of the graph node of a controlled math instruction.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./dsp_control.fpga_emu     (Linux)
     dsp_control.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./dsp_control.fpga         (Linux)
     ```

### Example of Output
```
PASSED: all kernel results are correct.
```

### Discussion

Feel free to experiment further with the tutorial code. You can:
 - Try various control options with global control and local control.
 - Try various datatypes and math operations that support DSP control.
