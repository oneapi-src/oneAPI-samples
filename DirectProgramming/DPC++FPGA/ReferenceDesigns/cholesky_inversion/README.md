# `Cholesky-based Matrix Inversion` Sample

This `Cholesky-based Matrix Inversion` sample is a reference design that demonstrates the use of Cholesky decomposition and inversion header libraries on 32x32 real matrices:

- `streaming_cholesky.hpp` - linear algebra header library implements the Cholesky decomposition of matrices with pipe interfaces.
- `streaming_cholesky_inversion.hpp` - linear algebra header library implements the inversion of matrices with pipe interfaces, based on the outputs of the cholesky decomposition implemented in `streaming_cholesky.hpp`. 

Both the decomposition and the inversion have template parameters that can be set to decompose/invert custom sized real or complex square matrices.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement high performance Cholesky matrix decomposition on field programmable gate arrays (FPGAs).
| Time to complete      | ~1 hr (excluding compile time)

## Purpose

This FPGA reference design demonstrates Cholesky-based matrix inversion, a common operation employed in linear algebra, as implemented in supplied header libraries.

The Cholesky decomposition takes a hermitian, positive-definite matrix $A$ (input) and computes the lower triangular matrix $L$ such that:

$LL^{\star} = A$
where $L^{\star}$ in the conjugate transpose of $L$.

Given this decomposition, the inverse of A can be computed as

$A^{-1} = (LL^{\star})^{-1} = (L^{\star})^{-1}L^{-1} = (L^{-1})^{\star} L^{-1}$

Therefore, in order to compute the inverse of A, one can: compute $LI$, the invert of $L$ (triangular matrix) and perform a matrix multiplication of the transpose of $LI$ by $LI$.

$A^{-1} = LI^{\star}LI$

Because $A^{-1}$  is going to be symmetric, one can also compute only half of the values.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler <br> Intel® FPGA Add-on for oneAPI Base Toolkit

### Performance

> **Note**: See the [Performance Disclaimers](#performance-disclaimers) section below.

| Device                                            | Throughput
|:---                                               |:---
| Intel® PAC with Intel Arria® 10 GX FPGA           | 215k matrices/s for real matrices of size 32x32
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) | 255k matrices/s for real matrices of size 32x32


## Key Implementation Details

In this reference design, the Cholesky decomposition algorithm is used to factor a real _n_ × _n_ matrix. The algorithm computes the vector dot product of two rows of the matrix. In our FPGA implementation, the dot product is computed in a loop over the row's _n_ elements. The loop is fully unrolled to maximize throughput. As a result, *n* real multiplication operations are performed in parallel on the FPGA, followed by sequential additions to compute the dot product result.

We use the compiler option `-fp-relaxed`, which permits the compiler to reorder floating point additions (i.e. to assume that floating point addition is commutative). The compiler uses this freedom to reorder the additions so that the dot product arithmetic can be optimally implemented using the FPGA's specialized floating point DSP (Digital Signal Processing) hardware.

With this optimization, our FPGA implementation requires _n_ DSPs to compute the real floating point dot product. The input matrix is also replicated two times in order to be able to read two full rows per cycle. The matrix size is constrained by the total FPGA DSP and RAM resources available.

The matrix inversion algorithm used in this reference design performs a Gaussian elimination to invert the triangular matrix _L_ obtained by the Cholesky decomposition. To do so, another _n_ DSPs are required to perform the associated dot-product. Finally, the matrix product of $LI^{\star}LI$ also requires _n_ DSPs.

A handful of extra DSPs are required to perform various arithmetic operations such as division, reciprocal square root, among others.

| Kernel                  | Description
|:---                     |:---
| `CholeskyDecomposition` | Implements a modified Cholesky–Banachiewicz decomposition algorithm.
| `CholeskyInversion`     | Implements the matrix inversion based on the CholeskyDecomposition's outputs.

To optimize the performance-critical loops in these algorithms, the design leverages concepts discussed in the following FPGA tutorials:

- **Triangular Loop Optimization** (triangular_loop)
- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Unrolling Loops** (loop_unroll)

The design uses the following key optimization techniques:
1. Traversing the matrix in a column fashion to increase the read-after-write (RAW) loop iteration distance.
2. Using two copies of the compute matrix in order to be able to read a full row and a full column per cycle.
3. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
4. Fully vectorizing the dot products using loop unrolling.
5. Using the compiler flag -Xsfp-relaxed to re-order floating point operations and allowing the inference of a specialized dot-product DSP. This further reduces the number of DSP blocks needed by the implementation, the overall latency, and pipeline depth.
6. Using an efficient memory banking scheme to generate high performance hardware (all local memories are single-read, single-write).
7. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.
8. Using the input matrices properties (hermitian positive matrices) to reduce the number of operations. For example, the (_LI_*) * _LI_ computation only requires to compute half of the output matrix as the result is symmetric.

### Source Code Breakdown

The following source files can be found in the `src/` sub-directory. 

| File                                | Description
|:---                                 |:---
|`cholesky_inversion_demo.cpp`        | Contains the `main()` function that generates the input matrices, calls the compute function and validates the results.
|`cholesky_inversion.hpp`             | Contains the compute function that calls the kernels.
|`memory_transfers.hpp`               | Contains functions to transfer matrices from/to the FPGA DDR with streaming interfaces.

For descriptions of `streaming_cholesky.hpp`, `streaming_cholesky_inversion.hpp`, `constexpr_math.hpp`, `memory_utils.hpp`, `metaprogramming_utils.hpp`, and `unrolled_loop.hpp` see the README in the `DirectProgramming/DPC++FPGA/include/` directory.

### Compiler Flags Used

| Flag                              | Description
|:---                               |:---
|`-Xshardware`                      | Target FPGA hardware (as opposed to FPGA emulator)
|`-Xsclock=<target fmax>MHz`        | The FPGA backend attempts to achieve <target fmax> MHz
|`-Xsfp-relaxed`                    | Allows the FPGA backend to re-order floating point arithmetic operations (for example, permit assuming $(a + b + c) == (c + a + b)$)
|`-Xsparallel=2`                    | Use 2 cores when compiling the bitstream through Quartus
|`-Xsseed`                          | Specifies the Quartus compile seed to yield slightly higher, possibly, fmax
|`-DMATRIX_DIMENSION`               | Specifies the number of rows/columns of the matrix
|`-DFIXED_ITERATIONS_DECOMPOSITION` | Used to set the ivdep safelen attribute for the performance critical triangular loop of the decomposition kernel
|`-DFIXED_ITERATIONS_INVERSION`     | Used to set the ivdep safelen attribute for the performance critical triangular loop of the inversion kernel
|`-DCOMPLEX`                        | Used to select between the complex and real QR decomposition (real is the default)

> **Note**: The values for `seed`, `FIXED_ITERATIONS_DECOMPOSITION`, `DFIXED_ITERATIONS_INVERSION`, `MATRIX_DIMENSION`, `COMPLEX` depend on the board being targeted.

### Additional Documentation

- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Cholesky-based Matrix Inversion` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
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

### Include Files

The include folder is at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

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
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
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
      The report resides at `cholesky_inversion_report.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
       ```
       make fpga
       ```

     (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/cholesky_inversion.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/cholesky_inversion.fpga.tar.gz).

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
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
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
      The report resides at `cholesky_inversion_report.a.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Cholesky-based Matrix Inversion` Program

### Configurable Parameters

| Parameter    | Description
|:---          |:---
| `<num>`      | (Optional) Argument to specify the number of times to repeat the inversion of 8 matrices. Its default value is `16` for the emulation flow and `819200` for the FPGA flow. Pass the parameter as an input value: `<program name> <num>`.

You can apply the Cholesky-based inversion to 8 matrices repeated a number of times optionally passed to the program. This step performs the following:

- Generates 8 random matrices (this number can be changed in cholesky_inversion_demo.cpp).
- Computes the Cholesky-based inversion on all matrices.
- Repeats the operation multiple times (optionally specified as the command line argument) to evaluate performance.

### On Linux

1. Run on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./cholesky_inversion.fpga_emu
   ```
2. Run on the FPGA device.
   ```
   ./cholesky_inversion.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   cholesky_inversion.fpga_emu.exe
   ```

### Build and Run the Samples on Intel® DevCloud (Optional)

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

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **24 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.

## Example Output

This example output shows typical results for 8 matrices inverted 819,200 times with each matrix consisting of **32x32** real numbers.

### Intel® PAC with Intel Arria® 10 GX FPGA
```
Device name: pac_a10 : Intel PAC Platform (pac_f100000)
Generating 8 random real matrices of size 32x32
Computing the Cholesky-based inversion of 8 matrices 819200 times
   Total duration:   30.442 s
Throughput: 215.281k matrices/s
Verifying results...

PASSED
```
### Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)
```
Device name: pac_s10 : Intel PAC Platform (pac_f100000)
Generating 8 random real matrices of size 32x32 
Computing the Cholesky-based inversion of 8 matrices 819200 times
 Total duration:   25.694 s
Throughput: 255.063k matrices/s
Verifying results...

PASSED
```

## Performance Disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of April 26, 2022 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software, or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

© Intel Corporation.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).