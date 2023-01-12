# `Matrix Multiply` Sample

This reference design demonstrates high-performance general matrix multiplication on an FPGA.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement a high performance FPGA version of a general matrix multiplication algorithm.
| Time to complete      | 1 hr (not including compile time)
| Category              | Reference Designs and End to End

## Purpose

This FPGA reference design demonstrates the multiplication of matrices, a common operation employed in linear algebra.

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
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

### Performance

Performance results are based on testing as of ???.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Intel® PAC with Intel® Arria® 10 GX FPGA          | ???
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) | ???


## Key Implementation Details

This algorithm multiplies a *m* × *n* matrix with a *n* × *p* matrix using a systolic array. The systolic array consists of a two-dimensional grid of *m* × *p* processing elements (PEs). Each PE is responsible for computing one element of the output matrix. Every cycle, it consumes data from neighboring PEs, and performs a multiply-accumulate. In our FPGA implementation, this is achieved by fully unrolling the loop over the PEs. In total, *mp* multiply-accumulates are performed in parallel on the FPGA.

The multiply-accumulate arithmetic can be optimally implemented using the FPGA's specialized floating point DSP (Digital Signal Processing) hardware. Our FPGA implementation requires *mp* DSPs to compute the multiply-accumulate over all PEs in one cycle. Thus, the matrix size is constrained by the total FPGA DSP resources available.

The design uses the `fpga_reg` attribute to insert additional pipelining registers along the path to each PE. This allows values to be passed from one PE to the next, which reduces the fanout caused by distributing the same value to all PEs in one row or column.

For larger matrices, where it is infeasible to generate a systolic array to compute the full matrix, we split the matrix into smaller tiles, each of which is computed sequentially using a systolic array corresponding to the size of the tile.

To optimize the performance-critical loop in its algorithm, the design leverages concepts discussed in the following FPGA tutorials:

- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Unrolling Loops** (loop_unroll)

The key optimization techniques used are as follows:

1. Fully unrolling the loop over the matrix product to instantiate a two-dimensional systolic array of PEs
2. Using an efficient memory banking scheme to generate high performance hardware.
3. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.

### Compiler Flags Used

| Flag                  | Description
|:---                   |:---
| `-Xshardware`         | Target FPGA hardware (as opposed to FPGA emulator)
| `-Xsclock=360MHz`     | The FPGA backend attempts to achieve 360 MHz
| `-Xsseed`             | Specifies the Intel® Quartus® compile seed, to yield slightly higher fmax

Additionaly, the cmake build system can be configured using the following parameters:

| cmake option              | Description
|:---                       |:---
| `-DSET_ROWS_A` | Specifies the number of rows of matrix A
| `-DSET_COMMON` | Specifies the number of columns of matrix A / rows of matrix B (these values are equal)
| `-DSET_COLS_B` | Specifies the number of columns of matrix B
| `-DSET_TILE_A` | Specifies the tile size used on matrix A
| `-DSET_TILE_B` | Specifies the tile size used on matrix B
| `-DSET_TILE_COMMON` | Specifies the tile size used on the common dimension between matrices A and B
| `-DSET_DEBUG` | Used to turn on/off debug information (printing out matrices)

## Build the `Matrix multiply` Design

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
2. Configure the build system for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```
   For **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```

   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      make fpga_sim
      ```

   3. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `matmul.report.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

   <!-- (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/matmul.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/matmul.fpga.tar.gz). -->

### On Windows*

>**Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.
2. Configure the build system for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   To compile for the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      nmake fpga_sim
      ```
   3. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `qrd_report.a.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Matrix Multiply` Design

### Configurable Parameters

| Argument  | Description
|:---       |:---
| `<num>`   | (Optional) Specifies the number of times to repeat the multiplication of a set of 8 matrices (only 1 matrix when running simulation). Its default value is **16** for the emulation and simulation flows, and **819200** for the FPGA flow.

You can perform the multiplication of the set of matrices repeatedly. This step performs the following:
- Generates the set of random matrices.
- Computes the product of the set of matrices.
- Repeats the multiplication multiple times (specified as a command line argument) to evaluate performance.

### On Linux

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./matmul.fpga_emu
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./qrd.fpga_sim
   ```

#### Run on FPGA

1. Run the sample on the FPGA device.
   ```
   ./matmul.fpga
   ```

### On Windows

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   matmul.fpga_emu.exe
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   qrd.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
#### Run on FPGA

1. Run the sample on the FPGA device.
   ```
   matmul.fpga.exe
   ```

## Example Output

Example output when running on **Intel® PAC with Intel® Arria® 10 GX FPGA** for the multiplication of 8 matrices 819200 times (each matrix consisting of 64x64 single-precision floating point numbers, computed using a systolic array of 8x8 PEs).

```
Device name: pac_a10 : Intel PAC Platform (pac_f000000)
 Matrix A size: 64 x 64 (tile: 8 x 64)
 Matrix B size: 64 x 64 (tile: 64 x 8)
 Systolic array size: 8 x 8 PEs
Running matrix multiplication of 8 matrices 819200 times
 Total duration: ??? s
Throughput: ???k matrices/s

PASSED
```

Example output when running on **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)** for the multiplication of 8 matrices 819200 times (each matrix consisting of 64x64 single-precision floating point numbers, computed using a systolic array of 8x8 PEs).

```
Device name: pac_s10 : Intel PAC Platform (pac_f100000)
 Matrix A size: 64 x 64 (tile: 8 x 64)
 Matrix B size: 64 x 64 (tile: 64 x 8)
 Systolic array size: 8 x 8 PEs
Running matrix multiplication of 8 matrices 819200 times
 Total duration: ??? s
Throughput: ???k matrices/s

PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).