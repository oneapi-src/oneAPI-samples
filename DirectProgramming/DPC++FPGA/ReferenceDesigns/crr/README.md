# CRR Binomial Tree Model for Option Pricing
A field programmable gate array (FPGA)-optimized reference design computing the Cox-Ross-Rubinstein (CRR) binomial tree model with Greeks for American exercise options.

## Purpose
This sample implements the Cox-Ross-Rubinstein (CRR) binomial tree model that is used in the finance field for American exercise options with five Greeks (delta, gamma, theta, vega and rho). The simple idea is to model all possible asset price paths using a binomial tree.

## Prerequisites
| Optimized for                     | Description
---                                 |---
| OS                                | Ubuntu* 18.04/20.04 <br>RHEL*/CentOS* 8 <br>SUSE* 15 <br>Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg; 10 GX FPGA; <br> Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX)
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br> Intel&reg; FPGA Add-On for oneAPI Base Toolkit

### Performance
> **Note**: Please refer to the [Performance Disclaimers](#performance-disclaimers) section below.

| Device                                         | Throughput
|:---                                            |:---
| Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA        | 118 assets/s
| Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX)      | 243 assets/s

### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.


## Key Implementation Details
### Design Inputs
This design reads inputs from the `ordered_inputs.csv` file. The inputs are:

| Input                             | Description
---                                 |---
| `n_steps`                         | Number of time steps in the binomial tree. The maximum `n_steps` in this design is 8189.
| `cp`                              | -1 or 1 represents put and call options, respectively.
| `spot`                            | Spot price of the underlying price.
| `fwd`                             | Forward price of the underlying price.
| `strike`                          | Exercise price of the option.
| `vol`                             | Percent volatility that the design reads as a decimal value.
| `df`                              | Discount factor to option expiry.
| `t`                               | Time, in years, to the maturity of the option.

### Design Outputs
This design writes outputs to the `ordered_outputs.csv` file. The outputs are:

| Output                            | Description
---                                 |---
| `value`                           | Option price
| `delta`                           | Measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price.
| `gamma`                           | Measures the rate of change in the `delta` with respect to changes in the underlying price.
| `vega`                            | Measures sensitivity to volatility.
| `theta`                           | Measures the sensitivity of the derivative's value to the passage of time.
| `rho`                             | Measures sensitivity to the interest of rate.

#### Design Correctness
This design tests the optimized FPGA code's correctness by comparing its output to a golden result computed on the CPU.

#### Design Performance
This design measures the FPGA performance to determine how many assets can be processed per second.

## Building the CRR Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Run Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see [Intel&reg; DevCloud for oneAPI Get Started](https://devcloud.intel.com/oneapi/documentation/base-toolkit/).

When compiling for FPGA hardware, it is recommended to increase the job timeout to **48h**.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see [Using Visual Studio Code with Intel&reg; oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
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
3. (Optional) As the above hardware compile may take several hours to complete, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/crr.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/crr.fpga.tar.gz).

### On Windows*

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
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

> **Note**: The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Troubleshooting
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See the [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information.

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, see [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).

## Running the Reference Design

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./crr.fpga_emu <input_file> [-o=<output_file>]                           (Linux)

     crr.fpga_emu.exe <input_file> [-o=<output_file>]                         (Windows)
     ```
 2. Run the sample on the FPGA device:
     ```
     ./crr.fpga <input_file> [-o=<output_file>]                               (Linux)
        
     crr.fpga.exe <input_file> [-o=<output_file>]                             (Windows)
     ```

### Application Parameters

| Argument                          | Description
---                                 |---
| `<input_file>`                    | Optional argument that provides the input data. The default file is `/data/ordered_inputs.csv`
| `-o=<output_file>`                | Optional argument that specifies the name of the output file. The default name of the output file is `ordered_outputs.csv`.

### Example of Output
```
============ Correctness Test =============
Running analytical correctness checks...
CPU-FPGA Equivalence: PASS

============ Throughput Test =============
Avg throughput: 66.2 assets/s
```

## Additional Design Information

### Source Code Explanation

| File                              | Description
---                                 |---
| `main.cpp`                        | Contains both host code and SYCL* kernel code.
| `CRR_common.hpp`                  | Header file for `main.cpp`. Contains the data structures needed for both host code and SYCL* kernel code.



### Backend Compiler Flags Used

| Flag                              | Description
---                                 |---
`-Xshardware`                       | Target FPGA hardware (as opposed to FPGA emulator)
`-Xsdaz`                            | Denormals are zero
`-Xsrounding=faithful`              | Rounds results to either the upper or lower nearest single-precision numbers
`-Xsparallel=2`                     | Uses 2 cores when compiling the bitstream through Quartus&reg;
`-Xsseed=2`                         | Uses seed 2 during Quartus&reg;, yields slightly higher f<sub>MAX</sub>

### Preprocessor Define Flags

| Flag                              | Description
---                                 |---
`-DOUTER_UNROLL=1`                  | Uses the value 1 for the constant OUTER_UNROLL, controls the number of CRRs that can be processed in parallel
`-DINNER_UNROLL=64`                 | Uses the value 64 for the constant INNER_UNROLL, controls the degree of parallelization within the calculation of 1 CRR
`-DOUTER_UNROLL_POW2=1`             | Uses the value 1 for the constant OUTER_UNROLL_POW2, controls the number of memory banks


> **Note**: The `Xsseed`, `DOUTER_UNROLL`, `DINNER_UNROLL` and `DOUTER_UNROLL_POW2` values differ depending on the board being targeted. More information about the unroll factors can be found in `/src/CRR_common.hpp`.

### Performance Disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase. For more complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).

Performance results are based on testing as of July 20, 2020 and may not reflect all publicly available security updates. See configuration disclosure for details. No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 20, 2020

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

&copy; Intel Corporation.

## Additional References

- [Khronous Group SYCL Resources](https://www.khronos.org/sycl/resources)

- [Wikipedia article on Binomial options pricing model](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

- [Wikipedia article on Greeks (finance)](https://en.wikipedia.org/wiki/Greeks_(finance))

- [GitHub repository for Intercept Layer for OpenCL&trade; Applications](https://github.com/intel/opencl-intercept-layer)

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
