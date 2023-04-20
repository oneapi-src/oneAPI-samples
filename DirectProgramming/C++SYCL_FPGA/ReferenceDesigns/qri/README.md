# `QRI` Sample

This `QRI` reference design demonstrates high performance QR-based inversion of real and complex matrices on FPGA.

| Area                 | Description
|:---                  |:---
| What you will learn  | Implementing a high performance FPGA version of the Gram-Schmidt QR decomposition to compute a matrix inversion.
| Time to complete     | 1 hr (not including compile time)
| Category             | Reference Designs and End to End

## Purpose

This FPGA reference design demonstrates QR-based inversion of real and complex matrices, a common operation employed in linear algebra.

The algorithms employed by the reference design are the **Gram-Schmidt process** QR decomposition algorithm and the thin **QR factorization** method. The original algorithm has been modified and optimized for performance on FPGAs in this implementation.

The inverse of the input matrix A: inv(A) is then computed as inv(R) * transpose(Q).

>**Note**: Read the *[QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition)* Wikipedia article for more information on the algorithms.

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

## Key Implementation Details

The QR-based matrix inversion algorithm factors a complex *n* × *n* matrix. The algorithm computes the QR decomposition of the input matrix. To do so, it computes the vector dot product of two columns of the matrix. In our FPGA implementation, the dot product is computed in a loop over the column's *n* elements. The loop is fully unrolled to maximize throughput. As a result, *n* complex multiplication operations are performed in parallel on the FPGA, followed by sequential additions to compute the dot product result. The computation of the inverse of R also requires a dot product in a loop over *n* elements. Finally, the final matrix multiplication also requires a dot product in a loop over *n* elements.

With this optimization, our FPGA implementation requires 4*_n_ DSPs to compute each of the complex floating point dot products or 2*_n_ DSPs in the real case. Thus, the matrix size is constrained by the total FPGA DSP resources available.

By default, the design is parameterized to process real floating-point matrices of size 32 × 32.

To optimize the performance-critical loop in its algorithm, the design leverages concepts discussed in the following FPGA tutorials:

- **Triangular Loop Optimization** (triangular_loop)
- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Unrolling Loops** (loop_unroll)
- **Shannonization** (shannonization)

The key optimization techniques used are as follows:
1. Refactoring the original Gram-Schmidt algorithm to merge two dot products of the QR decomposition into one, thus reducing the total number of dot products needed from three to two. This helps us reduce the DSPs required for the implementation.
2. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
3. Fully vectorizing the dot products using loop unrolling.
4. Using an efficient memory banking scheme to generate high performance hardware.
5. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.
6. Using the triangular loop optimization technique to maintain high throughput in triangular loops

### Compiler Flags Used

| Flag                      | Description
|:---                       |:---
| `-Xshardware`             | Target FPGA hardware (as opposed to FPGA emulator)
| `-Xsclock=330MHz`         | The FPGA backend attempts to achieve 330 MHz
| `-Xsparallel=2`           | Use 2 cores when compiling the bitstream through Intel® Quartus®
| `-Xsseed`                 | Specifies the Intel® Quartus® compile seed, to yield slightly higher fmax

Additionaly, the cmake build system can be configured using the following parameters:

| cmake option                  | Description
|:---                           |:---
| `-DSET_ROWS_COMPONENT`        | Specifies the number of rows of the matrix
| `-DSET_COLS_COMPONENT`        | Specifies the number of columns of the matrix
| `-DSET_FIXED_ITERATIONS_QRD`  | Used to set the ivdep safelen attribute for the performance critical triangular loop in the QR decomposition kernel
| `-DSET_FIXED_ITERATIONS_QRI`  | Used to set the ivdep safelen attribute for the performance critical triangular loop in the QR inversion kernel
| `-DSET_COMPLEX`               | Used to select between the complex and real QR decomposition/inversion

>**Note**: The values for `-Xsseed`, `-DSET_FIXED_ITERATIONS_QRD`, `-DSET_FIXED_ITERATIONS_QRI`, `-DSET_ROWS_COMPONENT`, `-DSET_COLS_COMPONENT` and `-DSET_COMPLEX` depend on the board being targeted.

## Build the `QRI` Sample

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
   3. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `qri_report/reports/report.html`.

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
   3. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `qri_report.a.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `QRI` Design

### Configurable Parameters

| Argument   | Description
|:---        |:---
| `<num>`    | (Optional) Specifies the number of times to repeat the inversion of a set of 8 matrices (only 1 matrix when running simulation). Its default value is **16** for the emulation, **1** for the simulation flow and **6553600** for the FPGA flow.

You can perform the QR-based inversion of the set of matrices repeatedly, as shown below. This step performs the following:

- Generates the set of random matrices.
- Computes the QR-based inversion of the set of matrices.
- Repeats the decomposition multiple times (specified as a command line argument) to evaluate performance.

### On Linux

#### Run on FPGA Emulator

1. Increase the amount of memory that the emulator runtime is permitted to allocate by exporting the `CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE` environment variable before running the executable.
2. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   export CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
   ./qri.fpga_emu
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./qri.fpga_sim
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./qri.fpga
   ```

### On Windows

#### Run on FPGA Emulator

1. Increase the amount of memory that the emulator runtime is permitted to allocate by setting the `CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE` environment variable before running the executable.
2. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   set CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
   qri.fpga_emu.exe
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   qri.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   qri.fpga.exe
   ```

## Example Output

Example output when running the emulator on 8 matrices (each consisting of 32 x 32 real numbers).

```
Device name: Intel(R) FPGA Emulation Device
Generating 8 random real matrices of size 32x32
Running QR inversion of 8 matrices 16 times
   Total duration:   0.50876 s
Throughput: 0.251592k matrices/s
Verifying results...
PASSED
```

Example output when running on **Intel® PAC with Intel® Arria® 10 GX FPGA** for 8 matrices (each consisting of 32 x 32 real numbers).

```
Device name: pac_a10 : Intel PAC Platform (pac_f100000)
Generating 8 random real matrices of size 32x32
Running QR inversion of 8 matrices 6553600 times
   Total duration:   267.352 s
Throughput: 196.104k matrices/s
Verifying results...
PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
