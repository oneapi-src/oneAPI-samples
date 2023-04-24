# `Matrix Multiply` Sample

This reference design demonstrates a systolic-array-based high-performance general matrix multiplication on an FPGA.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement a systolic-array-based high-performance FPGA version of a general matrix multiplication algorithm.
| Time to complete      | 1 hr (not including compile time)
| Category              | Reference Designs and End to End

## Purpose

This FPGA reference design demonstrates the multiplication of matrices, a common operation employed in linear algebra. Matrices *A* and *B* (inputs) are multiplied to produce matrix C. The matrix multiplication algorithm is implemented using a systolic array of Processing Elements (PEs), and has been optimized for performance on FPGAs.

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
| Hardware             | Intel® Agilex™ 7, Arria® 10, and Stratix® 10 FPGAs
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

Performance results are based on testing as of March 6, 2023.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Intel® PAC with Intel® Arria® 10 GX FPGA          | 75k matrices/s for single-precision floating-point matrices of size 64 * 64, computed using a systolic array of 8 * 8 PEs (64 DSPs)
| Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) | 90k matrices/s for single-precision floating-point matrices of size 64 * 64, computed using a systolic array of 8 * 8 PEs (64 DSPs)


## Key Implementation Details

The matrix multiplication algorithm multiplies two single-precision floating-point matrices *A* (*m × n*) and *B* (*n × p*) to produce matrix *C* (*m × p*). The algorithm is implemented using a systolic array approach, employing an array of *m<sub>PE</sub> × p<sub>PE</sub>* PEs, each of which is responsible for computing a dot product corresponding to an element of the output matrix. As a result, this implementation computes the matrix multiplication in tiles of *m<sub>PE</sub> × p<sub>PE</sub>* at a time.

<p align="center">
  <img src=assets/overview.png />
</p>

Each iteration, a PE consumes inputs from previous PEs in the same row or column and uses these data to compute a multiply-accumulate. It then passes the inputs along to the next PE. We orchestrate the flow of data through the systolic array such that each PE is responsible for computing one element of the output matrix. Using this approach, it takes a total of *n* iterations to compute one matrix product, where every iteration, we compute a partial sum for each of the elements of the output matrix.

<p align="center">
  <img src=assets/systolic_array.png />
</p>

Here, a single PE consists of logic to compute a floating-point multiply-accumulate and registers for storage of intermediate results. 

<p align="center">
  <img src=assets/PE.png />
</p>

In our FPGA implementation, the multiply-accumulate operations are computed in a nested loop over the PEs. The loop is fully unrolled to replicate the hardware for each PE. 

```c++
#pragma unroll
for (int row = 0; row < m_PE; row++) {
  #pragma unroll
  for (int col = 0; col < p_PE; col++) {
    fed_a[i] = fpga_reg(fed_a[i]);
    fed_b[j] = fpga_reg(fed_b[j]);
    accum[i][j] += fed_a[i] * fed_b[j];
    ...
  }
}
```

As a result, *m<sub>PE</sub> × p<sub>PE</sub>* multiply-accumulate operations are performed in parallel on the FPGA.  The multiply-accumulate arithmetic can be optimally implemented using the FPGA's specialized floating point DSP (Digital Signal Processing) hardware. When set to FP accum mode, we can perform this operation with an II of 1. With this optimization, our FPGA implementation requires *m<sub>PE</sub> × p<sub>PE</sub>* DSPs to compute all the floating-point multiply-accumulate operations. Thus, the matrix tile size is constrained by the total FPGA DSP resources available.

To optimize the performance-critical loop in its algorithm, the design leverages concepts discussed in the following FPGA tutorials:

- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Unrolling Loops** (loop_unroll)

The key optimization techniques used are as follows:

1. Fully unrolling the loop over the matrix product to instantiate an array of individual PEs in hardware.
2. Using the `fpga_reg` attribute to insert additional pipelining registers, a crucial step in the implementation of the systolic array structure of this design, which allows data to be passed from one PE to the next. The use of these registers both in the systolic array and in various other parts of the design improves the overall achievable frequency.
3. Using an efficient memory banking scheme to generate high performance hardware.

### Comparison with a naïve approach

Consider the following naïve implementation of a matrix multiplication kernel, where instead of implementing a systolic array, we unroll the loop computing the dot product. 
```c++
for (int row = 0; row < m; row++) {
  for (int col = 0; col < p; col++) {
    float dot_prod{0};
#pragma unroll
    for (int k = 0; k < n; k++) {
      dot_prod = fpga_reg(dot_prod) + a_load[row][k] * b_load[col][k];
    }
    mm_result[row][col] = dot_prod;
  }
}
```

This achieves the same objective of increasing throughput, though this design is not as scalable as the systolic array approach, as shown in the results below.

*Systolic array implementation*
| Matrix size | PE array size | DSP usage | Avg. fMAX
|:---         |:---           |:---       |:---
| 64x64       | 8x8           | 64        | 452.55 MHz
| 256x256     | 16x16         | 256       | 380.00 MHz

*Naïve implementation*
| Matrix size | Dot product size | DSP usage | Avg. fMAX
|:---         |:---              |:---       | :---   
| 64x64       | 64               | 64        | 435.28 MHz   
| 256x256     | 256              | 256       | 341.08 MHz


These results were gathered from targeting the Arria® 10 device family using the IP authoring flow, and the fMAX values were averaged across 6 seeds.

### Compiler Flags Used

| Flag              | Description
|:---               |:---
| `-Xshardware`     | Target FPGA hardware (as opposed to FPGA emulator)
| `-Xsclock=360MHz` | The FPGA backend attempts to achieve 360 MHz
| `-Xsseed`         | Specifies the Intel® Quartus® compile seed, to yield slightly higher fmax

Additionaly, the cmake build system can be configured using the following parameters:

| cmake option   | Description
|:---            |:---
| `-DSET_ROWS_A` | Specifies *m*, the number of rows of matrix A
| `-DSET_COMMON` | Specifies *n*, the number of columns of matrix A / rows of matrix B (these values are equal)
| `-DSET_COLS_B` | Specifies *p*, the number of columns of matrix B
| `-DSET_TILE_A` | Specifies *m<sub>PE</sub>*, the tile size used on matrix A
| `-DSET_TILE_B` | Specifies *p<sub>PE</sub>*, the tile size used on matrix B

>**Note**: The default values for `-Xsseed`, `-DSET_ROWS_A`, `-DSET_COMMON`, `-DSET_COLS_B`, `-DSET_TILE_A` and `-DSET_TILE_B` depend on the board being targeted.

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
2. Configure the build system for the Agilex™ 7 device family, which is the default.

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
      The report resides at `matmul.report.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Configure the build system for the Agilex™ 7 device family, which is the default.
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
      The report resides at `matmul_report.a.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Matrix Multiply` Design

### Configurable Parameters

| Argument  | Description
|:---       |:---
| `<num>`   | (Optional) Specifies the number of times to repeat the multiplication of a set of 8 matrices (only 1 matrix when running simulation). Its default value is **16** for the emulation flow, **1** for the simulation flow and **819200** for the FPGA flow.

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
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./matmul.fpga_sim
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
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
   matmul.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   matmul.fpga.exe
   ```

## Example Output

Example output when running on **Intel® PAC with Intel® Arria® 10 GX FPGA** for the multiplication of 8 matrices 819200 times (each matrix consisting of 64x64 single-precision floating point numbers, computed using a systolic array of 8x8 PEs).

```
Running on device: pac_a10 : Intel PAC Platform (pac_f100000)
 Matrix A size: 64 x 64 (tile: 8 x 64)
 Matrix B size: 64 x 64 (tile: 64 x 8)
 Systolic array size: 8 x 8 PEs
Running matrix multiplication of 2 matrices 819200 times
   Total duration:   21.8446 s
Throughput: 75.0025k matrices/s

PASSED
```

Example output when running on **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)** for the multiplication of 8 matrices 819200 times (each matrix consisting of 64x64 single-precision floating point numbers, computed using a systolic array of 8x8 PEs).

```
Running on device: pac_s10 : Intel PAC Platform (pac_f100000)
 Matrix A size: 64 x 64 (tile: 8 x 64)
 Matrix B size: 64 x 64 (tile: 64 x 8)
 Systolic array size: 8 x 8 PEs
Running matrix multiplication of 2 matrices 819200 times
   Total duration:   18.1147 s
Throughput: 90.4456k matrices/s

PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).