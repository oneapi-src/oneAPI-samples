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

This sample is part of the FPGA code samples.
It is categorized as a Tier 4 sample that demonstrates a reference design.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

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
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

> **Note**: You'll need a large FPGA part to be able to fit this design

### Performance

Performance results are based on testing as of July 20, 2020.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

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


#### Compiler Flags Used

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

## Build the `CRR Binomial Tree` Sample

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

### On Linux*

1. Change to the sample directory.
2. Configure the build system for the Agilex® 7 device family, which is the default.

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

### On Windows*

1. Change to the sample directory.
2. Configure the build system for the Agilex® 7 device family, which is the default.
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
   2. Generate the HTML performance report.
      ```
      nmake report
      ```
      The report resides at `<project name>.a.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `CRR Binomial Tree` Program

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./crr.fpga_emu <input_file> [-o=<output_file>]
   ```
   where:
   - `<input_file>` is an **optional** argument to specify the input data file name. The default input file is `/data/ordered_inputs.csv`.
   - `-o=<output_file>`  is an **optional** argument to  specify the name of the output file. The default name of the output file is `ordered_outputs.csv`.
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./crr.fpga_sim <input_file> [-o=<output_file>]
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
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
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   crr.fpga_sim.exe <input_file> [-o=<output_file>]
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   crr.fpga.exe <input_file> [-o=<output_file>]
   ```

## Example Output

```
============ Correctness Test =============
Running analytical correctness checks...
CPU-FPGA Equivalence: PASS

============ Throughput Test =============
Avg throughput: 66.2 assets/s
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
