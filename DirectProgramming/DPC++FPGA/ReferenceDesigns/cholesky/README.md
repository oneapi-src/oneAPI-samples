# Cholesky Decomposition of Matrices
The `streaming_cholesky.hpp` linear algebra header library implements the Cholesky decomposition of matrices with pipe interfaces. You can set the template parameters to decompose custom-sized real or complex square matrices.

This SYCL*-compliant reference design demonstrates the use of this Cholesky decomposition header library on 32x32 real matrices.

| Property                     | Description
|:---                          |:---
| What you will learn               | How to implement high performance matrix Cholesky decomposition on field programmable gate arrays (FPGAs).
| Time to complete                  | ~1 hr (excluding compile time)

## Purpose

This FPGA reference design demonstrates Cholesky decomposition, a common operation employed in linear algebra, through use of the `streaming_cholesky.hpp` header library.
The Cholesky decomposition takes a hermitian, positive-definite matrix _A_ (input) and computes the lower triangular matrix _L_ such that: _L_ * _L*_ = A, where _L*_ in the conjugate transpose of _L_.
The algorithm employed by the header library is a FPGA throughput optimized version of the Cholesky–Banachiewicz algorithm. (You can find information on the Cholesky decomposition and this algorithm can be found in the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) article on Wikipedia.)

Cholesky decomposition is used extensively in machine learning applications such as least-squares regressions and Monte-Carlo simulations.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04/20.04 <br>RHEL*/CentOS* 8 <br>SUSE* 15 <br>Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg; 10 GX FPGA <br> Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX) <br> Intel Xeon&reg; CPU E5-1650 v2 @ 3.50GHz (host machine)
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br> Intel&reg; FPGA Add-On for oneAPI Base Toolkit

### Performance
> **Note**: Please refer to the [Performance Disclaimers](#performance-disclaimers) section below.

| Device                                            | Throughput
|:---                                               |:---
| Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA           | 221k matrices/s for real matrices of size 32x32
| Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) | 214k matrices/s for real matrices of size 32x32

## Matrix Dimensions and FPGA resources

In this reference design, the Cholesky decomposition algorithm is used to factor a real _n_ × _n_ matrix. The algorithm computes the vector dot product of two rows of the matrix. In our FPGA implementation, the dot product is computed in a loop over the row's _n_ elements. The loop is fully unrolled to maximize throughput. As a result, *n* real multiplication operations are performed in parallel on the FPGA, followed by sequential additions to compute the dot product result.

We use the compiler option `-fp-relaxed`, which permits the compiler to reorder floating point additions (for example, to assume that floating point addition is commutative). The compiler uses this freedom to reorder the additions so that the dot product arithmetic can be optimally implemented using the FPGA's specialized floating point Digital Signal Processing (DSP) hardware.

With this optimization, our FPGA implementation requires _n_ DSPs to compute the real floating point dot product. The input matrix is also replicated two times in order to be able to read two full rows per cycle.
Thus, the matrix size is constrained by the total FPGA DSP and RAM resources available.


### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.


## Key Implementation Details
| Kernel            | Description
---                 |---
| Cholesky          | Implements a modified Cholesky–Banachiewicz Cholesky decomposition algorithm.

To optimize the performance-critical loop in its algorithm, the design leverages concepts discussed in the following FPGA tutorials:
- **Triangular Loop Optimization** (triangular_loop)
- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Unrolling Loops** (loop_unroll)

 The key optimization techniques used are as follows:
   1. Traversing the matrix in a column fashion to increase the read-after-write (RAW) loop iteration distance.
   2. Using two copies of the compute matrix in order to be able to read a full row and a full column per cycle.
   3. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
   4. Fully vectorizing the dot products using loop unrolling.
   5. Using the compiler flag -Xsfp-relaxed to re-order floating point operations and allowing the inference of a specialized dot-product DSP. This further reduces the number of DSP blocks needed by the implementation, the overall latency, and pipeline depth.
   6. Using an efficient memory banking scheme to generate high performance hardware (all local memories are single-read, single-write).
   7. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.

## Building the Reference Design

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Run Code Sample in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel&reg; oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to **24h**.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
1. Install the design into a directory `build` from the design directory by running `cmake`:

   ```
   mkdir build
   cd build
   ```

   If you are compiling for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:

   ```
   cmake ..
   ```

   If instead you are compiling for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following targets are provided, and they match the recommended development flow:

    - Compile for emulation (fast compile time, targets emulated FPGA device).

       ```
       make fpga_emu
       ```

    - Generate HTML performance report. Find the report in `cholesky_report.prj/reports/report.html`directory.

       ```
       make report
       ```

    - Compile for FPGA hardware (longer compile time, targets FPGA device).

       ```
       make fpga
       ```

3. (Optional) As the above hardware compile may take several hours to complete, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/cholesky.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/cholesky.fpga.tar.gz).

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
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:
   - Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   - Generate the optimization report:
     ```
     nmake report
     ```
   - Compile for FPGA hardware (longer compile time, targets FPGA device).
     ```
     nmake fpga
     ```

> **Note**: The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.


### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this Reference Design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).

## Running the Reference Design
You can apply the Cholesky decomposition to a number of matrices, as shown below. This step performs the following:
- Generates 8 random matrices (this number can be changed in cholesky_demo.cpp).
- Computes the Cholesky decomposition on all matrices.
- Repeats the operation multiple times (specified as the command line argument) to evaluate performance.

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
     ```
     ./cholesky.fpga_emu           (Linux)
     cholesky.fpga_emu.exe         (Windows)
     ```

2. Run the sample on the FPGA device.
     ```
     ./cholesky.fpga         (Linux)
     cholesky.fpga.exe       (Windows)
     ```
### Application Parameters

| Argument | Description
---        |---
| `<num>`  | Optional argument that specifies the number of times to repeat the decomposition of 8 matrices. Its default value is `16` for the emulation flow and '819200' for the FPGA flow.

### Example of Output

Example output when running on Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA for 8 matrices 819200 times (each matrix consisting of 32*32 real numbers):

```
Device name: pac_a10 : Intel PAC Platform (pac_f100000)
Generating 8 random real matrices of size 32x32 
Computing the Cholesky decomposition of 8 matrices 819200 times
   Total duration:   29.6193 s
Throughput: 221.261k matrices/s
Verifying results...

PASSED
```

Example output when running on Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) for the decomposition of 8 matrices 819200 times (each matrix consisting of 32*32 real numbers):

```
Device name: pac_s10 : Intel PAC Platform (pac_f100000)
Generating 8 random real matrices of size 32x32 
Computing the Cholesky decomposition of 8 matrices 819200 times
   Total duration:   30.58 s
Throughput: 214.31k matrices/s
Verifying results...

PASSED
```

## Additional Design Information

### Source Code Breakdown
The following source files can be found in the `src/` sub-directory. 

| File                                | Description
|:---                                 |:---
|`cholesky_demo.cpp`                  | Contains the `main()` function which generates the input matrices, calls the compute function and validates the results.
|`cholesky.hpp`                       | Contains the compute function that calls the kernels.
|`memory_transfers.hpp`               | Contains functions to transfer matrices from/to the FPGA DDR with streaming interfaces.

For `constexpr_math.hpp`, `memory_utils.hpp`, `metaprogramming_utils.hpp`, and `unrolled_loop.hpp` see the README in the `DirectProgramming/DPC++FPGA/include/` directory.

### Compiler Flags Used

| Flag | Description
---    |---
`-Xshardware` | Target FPGA hardware (as opposed to FPGA emulator)
`-Xsclock=<target fmax>MHz` | The FPGA backend attempts to achieve <target fmax> MHz
`-Xsfp-relaxed` | Allows the FPGA backend to re-order floating point arithmetic operations (for example, permit assuming $(a + b + c) == (c + a + b)$ )
`-Xsparallel=2` | Use 2 cores when compiling the bitstream through Quartus
`-Xsseed` | Specifies the Quartus compile seed, to potentially yield slightly higher fmax
`-DMATRIX_DIMENSION` | Specifies the number of rows/columns of the matrix
`-DFIXED_ITERATIONS` | Used to set the ivdep safelen attribute for the performance critical triangular loop
`-DCOMPLEX` | Used to select between the complex and real QR decomposition (real is the default)

> **Note**: The values for `seed`, `FIXED_ITERATIONS`, `MATRIX_DIMENSION`, `COMPLEX` are set according to the board being targeted.

## Performance Disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).

Performance results are based on testing as of February 25, 2022 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

&copy; Intel Corporation.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
