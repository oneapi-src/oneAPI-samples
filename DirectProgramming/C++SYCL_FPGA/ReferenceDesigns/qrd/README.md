# `QRD` Sample

The `QRD` reference design demonstrates high-performance QR decomposition of real and complex matrices on an FPGA.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement a high performance FPGA version of QR decomposition algorithm.
| Time to complete      | 1 hr (not including compile time)
| Category              | Reference Designs and End to End

## Purpose

This FPGA reference design demonstrates QR decomposition of matrices of complex/real numbers, a common operation employed in linear algebra. Matrix *A* (input) is decomposed into a product of an orthogonal matrix *Q* and an upper triangular matrix *R*.

The algorithms employed by the reference design are the **Gram-Schmidt process** QR decomposition algorithm and the thin **QR factorization** method. The original algorithm has been modified and optimized for performance on FPGAs in this implementation.

QR decomposition is used extensively in signal processing applications such as beamforming, multiple-input multiple-output (MIMO) processing, and Space Time Adaptive Processing (STAP).

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

Performance results are based on testing as of July 29, 2020.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Intel® PAC with Intel® Arria® 10 GX FPGA          | 24k matrices/s for complex matrices of size 128 * 128
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) | 7k matrices/s for complex matrices of size 256 * 256


## Key Implementation Details

The QR decomposition algorithm factors a complex _m_ × _n_ matrix, where _m_ ≥ _n_. The algorithm computes the vector dot product of two columns of the matrix. In our FPGA implementation, the dot product is computed in a loop over the column's _m_ elements. The loop is unrolled fully to maximize throughput. The *m* complex multiplication operations are performed in parallel on the FPGA followed by sequential additions to compute the dot product result.

The design uses the `-fp-relaxed` option, which permits the compiler to reorder floating point additions (to assume that floating point addition is commutative). The compiler reorders the additions so that the dot product arithmetic can be optimally implemented using the specialized floating point DSP (Digital Signal Processing) hardware in the FPGA.

With this optimization, our FPGA implementation requires 4*m* DSPs to compute the complex floating point dot product or 2*m* DSPs for the real case. The matrix size is constrained by the total FPGA DSP resources available.

By default, the design is parameterized to process 128 × 128 matrices when compiled targeting Intel® PAC with Intel Arria® 10 GX FPGA. It is parameterized to process 256 × 256 matrices when compiled targeting Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), a larger device; however, the design can process matrices from 4 x 4 to 512 x 512.

To optimize the performance-critical loop in its algorithm, the design leverages concepts discussed in the following FPGA tutorials:

- **Triangular Loop Optimization** (triangular_loop)
- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Unrolling Loops** (loop_unroll)

The key optimization techniques used are as follows:

1. Refactoring the original Gram-Schmidt algorithm to merge two dot products into one, reducing the total number of dot products needed to three from two. This helps us reduce the DSPs required for the implementation.
2. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
3. Fully vectorizing the dot products using loop unrolling.
4. Using the compiler flag -Xsfp-relaxed to re-order floating point operations and allowing the inference of a specialized dot-product DSP. This further reduces the number of DSP blocks needed by the implementation, the overall latency, and pipeline depth.
5. Using an efficient memory banking scheme to generate high performance hardware.
6. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.

### Compiler Flags Used

| Flag                  | Description
|:---                   |:---
| `-Xshardware`         | Target FPGA hardware (as opposed to FPGA emulator)
| `-Xsclock=360MHz`     | The FPGA backend attempts to achieve 360 MHz
| `-Xsfp-relaxed`       | Allows the FPGA backend to re-order floating point arithmetic operations (e.g. permit assuming (a + b + c) == (c + a + b) )
| `-Xsparallel=2`       | Use 2 cores when compiling the bitstream through Intel® Quartus®
| `-Xsseed`             | Specifies the Intel® Quartus® compile seed, to yield slightly higher fmax

Additionaly, the cmake build system can be configured using the following parameters:

| cmake option              | Description
|:---                       |:---
| `-DSET_ROWS_COMPONENT`    | Specifies the number of rows of the matrix
| `-DSET_COLS_COMPONENT`    | Specifies the number of columns of the matrix
| `-DSET_FIXED_ITERATIONS`  | Used to set the ivdep safelen attribute for the performance critical triangular loop
| `-DSET_COMPLEX`           | Used to select between the complex and real QR decomposition (complex is the default)

>**Note**: The values for `seed`, `-DSET_FIXED_ITERATIONS`, `-DSET_ROWS_COMPONENT`, `-DSET_COLS_COMPONENT` and `-DSET_COMPLEX` depend on the board being targeted.

## Build the `QRD` Design

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
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `qrd_report/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

   (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/qrd.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/qrd.fpga.tar.gz).

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
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `qrd_report.a.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `QRD` Design

### Configurable Parameters

| Argument  | Description
|:---       |:---
| `<num>`   | (Optional) Specifies the number of times to repeat the decomposition of 8 matrices. Its default value is **16** for the emulation flow and **819200** for the FPGA flow.

You can perform the QR decomposition of 8 matrices repeatedly. This step performs the following:
- Generates 8 random matrices.
- Computes the QR decomposition of the 8 matrices.
- Repeats the decomposition multiple times (specified as a command line argument) to evaluate performance.

### On Linux

#### Run on FPGA Emulator

1. Increase the amount of memory that the emulator runtime is permitted to allocate by exporting the `CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE` environment variable before running the executable.
2. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   export CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
   ./qrd.fpga_emu
   ```
#### Run on FPGA

1. Run the sample on the FPGA device.
   ```
   ./qrd.fpga
   ```

### On Windows

#### Run on FPGA Emulator

1. Increase the amount of memory that the emulator runtime is permitted to allocate by setting the `CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE` environment variable before running the executable.
2. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   set CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
   qrd.fpga_emu.exe
   ```
#### Run on FPGA

1. Run the sample on the FPGA device.
   ```
   qrd.fpga.exe
   ```

## Example Output

Example Output when running on **Intel® PAC with Intel® Arria® 10 GX FPGA** for 8 matrices 819200 times (each matrix consisting of 128x128 complex numbers).

```
Device name: pac_a10 : Intel PAC Platform (pac_f000000)
Generating 8 random complex matrices of size 128x128 
Running QR decomposition of 8 matrices 819200 times
 Total duration:   268.733 s
Throughput: 24.387k matrices/s
Verifying results...
PASSED
```

Example output when running on **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)** for the decomposition of 8 matrices 819200 times (each matrix consisting of 256x256 complex numbers).

```
Device name: pac_s10 : Intel PAC Platform (pac_f100000)
Generating 8 random complex matrices of size 256x256 
Running QR decomposition of 8 matrices 819200 times
 Total duration:   888.077 s
Throughput: 7.37954k matrices/s
Verifying results...
PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
