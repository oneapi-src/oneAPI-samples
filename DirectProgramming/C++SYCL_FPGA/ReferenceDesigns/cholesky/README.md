# `Cholesky Decomposition` of Matrices Sample

This SYCL*-compliant reference design demonstrates the use of this Cholesky decomposition header library for 32x32 real matrices on field programmable gate arrays (FPGAs).

| Area                    | Description
|:---                     |:---
| What you will learn     | How to implement high performance matrix Cholesky decomposition on FPGAs.
| Time to complete        | ~1 hr (excluding compile time)

## Purpose

The `Cholesky Decomposition` sample includes the `streaming_cholesky.hpp` linear algebra header library, which implements the Cholesky decomposition of matrices with pipe interfaces.

Cholesky decomposition is used extensively in machine learning applications such as least-squares regressions and Monte-Carlo simulations.

This FPGA reference design demonstrates Cholesky decomposition, a common operation employed in linear algebra, through use of the `streaming_cholesky.hpp` header library. The Cholesky decomposition takes a hermitian, positive-definite matrix _A_ (input) and computes the lower triangular matrix _L_ such that: _L_ * _L*_ = A, where _L*_ in the conjugate transpose of _L_.
The algorithm employed by the header library is an FPGA throughput optimized version of the Cholesky–Banachiewicz algorithm. (You can find information on the Cholesky decomposition and this algorithm can be found in the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) article on Wikipedia.)

You can set the template parameters to decompose custom-sized real or complex square matrices.

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

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware           | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software           | Intel® oneAPI DPC++/C++ Compiler

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

Performance results are based on testing as of February 25, 2022.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Intel® PAC with Intel Arria® 10 GX FPGA           | 221k matrices/s for real matrices of size 32x32
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) | 214k matrices/s for real matrices of size 32x32

## Key Implementation Details

The kernel is Cholesky, which implements a modified Cholesky–Banachiewicz Cholesky decomposition algorithm. To optimize the performance-critical loop in the algorithm, the design uses the several concepts discussed in the FPGA tutorial.

- **Triangular Loop Optimization** (triangular_loop)
- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Unrolling Loops** (loop_unroll)

The following list shows the key optimization techniques included in the reference design.
1. Traversing the matrix in a column fashion to increase the read-after-write (RAW) loop iteration distance.
2. Using two copies of the compute matrix to read a full row and a full column per cycle.
3. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This approach enables the ability to generate a design that is pipelined efficiently.
4. Fully vectorizing the dot products using loop unrolling.
5. Using an efficient memory banking scheme to generate high performance hardware (all local memories are single-read, single-write).
6. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.

### Matrix Dimensions and FPGA Resources

In this reference design, the Cholesky decomposition algorithm is used to factor a real _n_ × _n_ matrix. The algorithm computes the vector dot product of two rows of the matrix. In our FPGA implementation, the dot product is computed in a loop over the _n_ elements in the row. The loop is fully unrolled to maximize throughput, so *n* real multiplication operations are performed in parallel on the FPGA and followed by sequential additions to compute the dot product result.

With this optimization, our FPGA implementation requires _n_ DSPs to compute the real floating point dot product. The input matrix is also replicated two times in order to be able to read two full rows per cycle. The matrix size is constrained by the total FPGA DSP and RAM resources available.

### Compiler Flags Used

| Flag                        | Description
|:---                         |:---
|`-Xshardware`                | Target FPGA hardware (as opposed to FPGA emulator)
|`-Xsclock=<target fmax>MHz`  | The FPGA backend attempts to achieve <target fmax> MHz
|`-Xsparallel=2`              | Use 2 cores when compiling the bitstream through Quartus
|`-Xsseed`                    | Specifies the Quartus compile seed, to potentially yield slightly higher fmax

Additionaly, the cmake build system can be configured using the following parameters:

| cmake option                | Description
|:---                         |:---
|`-DSET_MATRIX_DIMENSION`     | Specifies the number of rows/columns of the matrix
|`-DSET_FIXED_ITERATIONS`     | Used to set the ivdep safelen attribute for the performance critical triangular loop
|`-DSET_COMPLEX`              | Used to select between the complex and real QR decomposition (real is the default)

> **Note**: The values for `-Xsseed`, `-DSET_MATRIX_DIMENSION`, `-DSET_FIXED_ITERATIONS`, and `-DSET_COMPLEX` depend on the board being targeted.

### Source Code

These source files are in the sample `src` directory.

| File                     | Description
|:---                      |:---
|`cholesky_demo.cpp`       | Contains the `main()` function which generates the input matrices, calls the compute function, and validates the results.
|`cholesky.hpp`            | Contains the compute function that calls the kernels.
|`memory_transfers.hpp`    | Contains functions to transfer matrices from/to the FPGA DDR with streaming interfaces.

For `constexpr_math.hpp`, `memory_utils.hpp`, `metaprogramming_utils.hpp`, and `unrolled_loop.hpp` see the README in the `DirectProgramming/C++SYCL_FPGA/include/` directory.

## Build the `Cholesky Decomposition` Reference Design

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
      The report resides at `cholesky_report.prj/reports/report.html`.

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
   1. Compile for emulation (fast compile time, targets emulated FPGA device):
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
   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Cholesky Decomposition` Executable

### Configurable Parameters

| Argument        | Description
|:---             |:---
| `<num>`         | (Optional) Specifies the number of times to repeat the decomposition of 8 matrices. The default value is `16` for the emulation flow and `819200` for the FPGA flow.

You can apply the Cholesky decomposition to a number of matrices, as shown below. This step performs the following:
- Generates 8 random matrices. You can change the number in the `cholesky_demo.cpp` source file.
- Computes the Cholesky decomposition on all matrices.
- Repeats the operation multiple times (specified as the command line argument) to evaluate performance.

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./cholesky.fpga_emu
   ```
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./cholesky.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./cholesky.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   cholesky.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   cholesky.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   cholesky.fpga.exe
   ```

## Example Output

These examples show output when running decomposition of 8 matrices 819,200 times with each matrix consisting of 32x32 real numbers.

### Example Output for Intel® PAC with Intel Arria® 10 GX FPGA

```
Device name: pac_a10 : Intel PAC Platform (pac_f100000)
Generating 8 random real matrices of size 32x32
Computing the Cholesky decomposition of 8 matrices 819200 times
   Total duration:   29.6193 s
Throughput: 221.261k matrices/s
Verifying results...

PASSED
```

### Example Output for Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)

```
Device name: pac_s10 : Intel PAC Platform (pac_f100000)
Generating 8 random real matrices of size 32x32
Computing the Cholesky decomposition of 8 matrices 819200 times
   Total duration:   30.58 s
Throughput: 214.31k matrices/s
Verifying results...

PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
