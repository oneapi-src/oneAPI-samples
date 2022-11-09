# `CRR Binomial Tree` Sample

The `CRR Binomial Tree` sample demonstrated a Cox-Ross-Rubinstein (CRR) binomial tree model using five Greeks for American option pricing and exercising in the form of a field programmable gate array (FPGA)-optimized reference design.

| Optimized for         | Description
|:---                   |:---
| What you will learn   | How to implement a Cox-Ross-Rubinstein (CRR) binomial tree for an FPGA
| Time to complete      | ~1 hr (excluding compile time)
| Category              | Reference Designs and End to End


## Purpose

This sample implements the Cox-Ross-Rubinstein (CRR) binomial tree model that is used in the finance field for American exercise options with five [Greeks](https://en.wikipedia.org/wiki/Greeks_(finance)) (delta, gamma, theta, vega, and rho). The code demonstrates how to model all possible asset price paths using a binomial tree.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler <br> Intel® FPGA Add-on for oneAPI Base Toolkit

### Performance

> **Note**: Please refer to the [Performance Disclaimers](#performance-disclaimers) section below.

| Device                                              | Throughput
|:---                                                 |:---
| Intel® PAC with Intel Arria® 10 GX FPGA             | 118 assets/s
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)   | 243 assets/s


## Key Implementation Details

### Design Inputs

This design reads inputs from the `ordered_inputs.csv` file. The inputs parameters are listed in the table.

| Input          | Description
|:---            |:---
| `n_steps`      | Number of time steps in the binomial tree. The maximum `n_steps` in this design is **8189**.
| `cp`           | -1 or 1 represents put and call options, respectively.
| `spot`         | Spot price of the underlying price.
| `fwd`          | Forward price of the underlying price.
| `strike`       | Exercise price of the option.
| `vol`          | Percent volatility that the design reads as a decimal value.
| `df`           | Discount factor to option expiry.
| `t`            | Time, in years, to the maturity of the option.

### Design Outputs

This design writes outputs to the `ordered_outputs.csv` file. The outputs are:

| Output         | Description
|:---            |:---
| `value`        | Option price
| `delta`        | Measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price.
| `gamma`        | Measures the rate of change in the `delta` with respect to changes in the underlying price.
| `vega`         | Measures sensitivity to volatility.
| `theta`        | Measures the sensitivity of the derivative's value to the passage of time.
| `rho`          | Measures sensitivity to the interest of rate.

### Design Correctness

This design tests the optimized FPGA code's correctness by comparing its output to a golden result computed on the CPU.

### Design Performance

This design measures the FPGA performance to determine how many assets can be processed per second.

### Additional Design Information

#### Source Code Explanation

| File                     | Description
|:---                      |:---
| `main.cpp`               | Contains both host code and SYCL* kernel code.
| `CRR_common.hpp`         | Header file for `main.cpp`. Contains the data structures needed for both host code and SYCL* kernel code.


#### Backend Compiler Flags Used

| Flag                    | Description
|:---                     |:---
|`-Xshardware`            | Target FPGA hardware (as opposed to FPGA emulator)
|`-Xsdaz`                 | Denormals are zero
|`-Xsrounding=faithful`   | Rounds results to either the upper or lower nearest single-precision numbers
|`-Xsparallel=2`          | Uses 2 cores when compiling the bitstream through Quartus®
|`-Xsseed=2`              | Uses seed 2 during Quartus®, yields slightly higher f<sub>MAX</sub>

#### Preprocessor Define Flags

| Flag                    | Description
|:---                     |:---
|`-DOUTER_UNROLL=1`       | Uses the value 1 for the constant OUTER_UNROLL, controls the number of CRRs that can be processed in parallel
|`-DINNER_UNROLL=64`      | Uses the value 64 for the constant INNER_UNROLL, controls the degree of parallelization within the calculation of 1 CRR
|`-DOUTER_UNROLL_POW2=1`  | Uses the value 1 for the constant OUTER_UNROLL_POW2, controls the number of memory banks

> **Note**: The `Xsseed`, `DOUTER_UNROLL`, `DINNER_UNROLL`, and `DOUTER_UNROLL_POW2` values differ depending on the board being targeted. You can find more information about the unroll factors in `/src/CRR_common.hpp`.


### Additional Documentation

- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `CRR Binomial Tree` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Change to the sample directory.

2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
       ```
       make fpga_emu
       ```
   2. Generate the HTML performance report.
       ```
       make report
       ```
      The report resides at `<project name>/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
       ```
       make fpga
       ```

      (Optional) As the above hardware compile may take several hours to complete, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/crr.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/crr.fpga.tar.gz).

### On Windows*

> **Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.

2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the HTML performance report.
      ```
      nmake report
      ```
      The report resides at `<project name>.a.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `CRR Binomial Tree` Program

### On Linux

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    ./crr.fpga_emu <input_file> [-o=<output_file>]
    ```
    where:
    - `<input_file>` is an **optional** argument to specify the input data file name. The default input file is `/data/ordered_inputs.csv`.
    - `-o=<output_file>`  is an **optional** argument to  specify the name of the output file. The default name of the output file is `ordered_outputs.csv`.

 2. Run the sample on the FPGA device.
    ```
    ./crr.fpga <input_file> [-o=<output_file>]
    ```

### On Windows

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    crr.fpga_emu.exe <input_file> [-o=<output_file>]
    ```
    where:
    - `<input_file>` is an **optional** argument to specify the input data file name. The default input file is `/data/ordered_inputs.csv`.
    - `-o=<output_file>`  is an **optional** argument to  specify the name of the output file. The default name of the output file is `ordered_outputs.csv`.

 2. Run the sample on the FPGA device.
    ```
    crr.fpga.exe <input_file> [-o=<output_file>]
    ```

### Build and Run the Sample on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

Use the Linux instructions to build and run the program.

You can specify an FPGA runtime node using a single line script similar to the following example.

```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:fpga_runtime:ppn=2` (lower case L) assigns one full node.
- `-d .` makes the current folder as the working directory for the task.


  |Available Nodes           |Command Options
  |:---                      |:---
  |FPGA Compile Time         |`qsub -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime (Arria 10)   |`qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d .`
  |FPGA Runtime (Stratix 10) |`qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d .`
  |GPU	                    |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	                    |`qsub -l nodes=1:xeon:ppn=2 -d .`

>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **48 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.

## Example Output

```
============ Correctness Test =============
Running analytical correctness checks...
CPU-FPGA Equivalence: PASS

============ Throughput Test =============
Avg throughput: 66.2 assets/s
```

### Performance Disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase. For more complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).

Performance results are based on testing as of July 20, 2020 and may not reflect all publicly available security updates. See configuration disclosure for details. No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 20, 2020.

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

© Intel Corporation.

## Additional References

- [Wikipedia article on Binomial options pricing model](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

- [GitHub repository for Intercept Layer for OpenCL™ Applications](https://github.com/intel/opencl-intercept-layer)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
