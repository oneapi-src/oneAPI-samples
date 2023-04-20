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

### Performance

Performance results are based on testing as of April 26, 2022.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Intel® PAC with Intel Arria® 10 GX FPGA           | 215k matrices/s for real matrices of size 32x32
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) | 255k matrices/s for real matrices of size 32x32

## Key Implementation Details

In this reference design, the Cholesky decomposition algorithm is used to factor a real _n_ × _n_ matrix. The algorithm computes the vector dot product of two rows of the matrix. In our FPGA implementation, the dot product is computed in a loop over the row's _n_ elements. The loop is fully unrolled to maximize throughput. As a result, *n* real multiplication operations are performed in parallel on the FPGA, followed by sequential additions to compute the dot product result.

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
5. Using an efficient memory banking scheme to generate high performance hardware (all local memories are single-read, single-write).
6. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.
7. Using the input matrices properties (hermitian positive matrices) to reduce the number of operations. For example, the (_LI_*) * _LI_ computation only requires to compute half of the output matrix as the result is symmetric.

### Source Code Breakdown

The following source files can be found in the `src/` sub-directory.

| File                                | Description
|:---                                 |:---
|`cholesky_inversion_demo.cpp`        | Contains the `main()` function that generates the input matrices, calls the compute function and validates the results.
|`cholesky_inversion.hpp`             | Contains the compute function that calls the kernels.
|`memory_transfers.hpp`               | Contains functions to transfer matrices from/to the FPGA DDR with streaming interfaces.

For descriptions of `streaming_cholesky.hpp`, `streaming_cholesky_inversion.hpp`, `constexpr_math.hpp`, `memory_utils.hpp`, `metaprogramming_utils.hpp`, and `unrolled_loop.hpp` see the README in the `DirectProgramming/C++SYCL_FPGA/include/` directory.

### Compiler Flags Used

| Flag                              | Description
|:---                               |:---
|`-Xshardware`                      | Target FPGA hardware (as opposed to FPGA emulator)
|`-Xsclock=<target fmax>MHz`        | The FPGA backend attempts to achieve <target fmax> MHz
|`-Xsparallel=2`                    | Use 2 cores when compiling the bitstream through Quartus
|`-Xsseed`                          | Specifies the Quartus compile seed to yield slightly higher, possibly, fmax

Additionaly, the cmake build system can be configured using the following parameters:

| cmake option                          | Description
|:---                                   |:---
|`-DSET_MATRIX_DIMENSION`               | Specifies the number of rows/columns of the matrix
|`-DSET_FIXED_ITERATIONS_DECOMPOSITION` | Used to set the ivdep safelen attribute for the performance critical triangular loop of the decomposition kernel
|`-DSET_FIXED_ITERATIONS_INVERSION`     | Used to set the ivdep safelen attribute for the performance critical triangular loop of the inversion kernel
|`-DSET_COMPLEX`                        | Used to select between the complex and real QR decomposition (real is the default)

> **Note**: The values for `-Xsseed`, `-DSET_FIXED_ITERATIONS_DECOMPOSITION`, `-DSET_FIXED_ITERATIONS_INVERSION`, `-DMATRIX_DIMENSION`, `-DSET_COMPLEX` depend on the board being targeted.

## Build the `Cholesky-based Matrix Inversion` Program

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
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      make fpga_sim
      ```
   3. Generate the HTML performance report.
      ```
      make report
      ```
      The report resides at `cholesky_inversion_report.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
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
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      nmake fpga_sim
      ```
   3. Generate the HTML performance report.
      ```
      nmake report
      ```
      The report resides at `cholesky_inversion_report.a.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Cholesky-based Matrix Inversion` Executable

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
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./cholesky_inversion.fpga_sim
   ```
3. Run on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./cholesky_inversion.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   cholesky_inversion.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   cholesky_inversion.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   cholesky_inversion.fpga.exe
   ```

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

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
