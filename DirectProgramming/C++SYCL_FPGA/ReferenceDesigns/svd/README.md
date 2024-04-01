# `Singular Value Decomposition (SVD)` Sample
The `SVD` reference design demonstrates a singular value decomposition implementation for real matrices on an FPGA.


| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement an FPGA version of singular value decomposition.
| Time to complete      | 1 hr (not including compile time)
| Category              | Reference Designs and End to End

## Purpose
This FPGA reference design demonstrates the Singular Value Decomposition (SVD) of real matrices. SVD is a common linear algebra factorization used for application such as matrix pseudoinverse and solving homogeneous linear equations.

SVD computes factorization:
```math
A = USV^T
```
Where U and V are made of (left and right) singular vectors and are both orthogonal. And S is a diagonal matrix consisted of singular values corresponding to singular vectors in U and V.

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
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
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

Performance results are based on testing as of March 20st, 2024.
> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Terasic DE10-Agilex Development Board             | 17.4k matrices/s for matrices of size 8 x 16

## Key Implementation Details
This SVD design consisted of 4 computation kernels in dark blue below, as well as several memory access kernels to handle input and output (in light blue). These kernels are connected through inter-kernel pipes and input/output through unified shared memory (USM).

![](assets/SVD.svg)

### Input covariance matrix computation
The covariance computation in this design is the same as used in the [PCA](../pca/README.md) reference design except without standardization.

The covariance of input A is equal to A multiplies transpose of A:
```math
Cov(A) = AA^T
```
Therefore this kernel only performs a matrix multiplication by blocks.

The implementation details of this computation is explained [here](../pca/README.md#input-matrix-standardization-and-covariance-matrix-computation) in the PCA sample.

### Eigen values and Eigen vectors computation
We are reusing the `fpga_linalg::StreamingEigen` design from the PCA sample to compute eigen values and eigen vectors of the input covariance matrix. Theses eigen values and eigen vectors are going to be used to construct the final outputs of SVD.

This kernel also produce an output `rank_deficent_flag` to indicate if the input matrix is not linearly independent. If the flag sets to 1, the result might be incorrect.

The [sources code](../../include/streaming_eigen.hpp) for the design can be found in the include directory. An detailed explanation of the algorithm of computing eigen values can be found [here](../pca/README.md#eigen-values-and-eigen-vectors-computation) in the PCA sample.

### Construct output from Eigen values and Eigen vectors
The final outputs of SVD can be constructed from the eigen values and eigen vectors of the input covariance. 

Consider an SVD input A of size m x n.

#### V Matrix
The right singular vectors (the V matrix) can be produced by simply copy the eigen vectors , and should be of size n x n:
```math
V = [v_0, v_1, v_2, ...,v_n] 
```

#### S Matrix
The singular value matrix (the S matrix) can be constructed as an m x n diagonal matrix (all but the diagonal are zeros) where the diagonal are square root of each eigen values:
```math
S =
\begin{bmatrix}
\sqrt{\lambda_{0}} & 0                  & 0                  & ... & 0 \\
0                  & \sqrt{\lambda_{1}} & 0                  & ... & 0 \\
0                  & 0                  & \sqrt{\lambda_{2}} & ... & 0 \\
...                & ...                & ...                & ... & 0 \\
0                  & 0                  & 0                  & 0   & \sqrt{\lambda_{n}} \\
0                  & 0                  & 0                  & 0   & 0
\end{bmatrix} 
```

#### U Matrix
The left singular vector (the U matrix) cannot be directly constructed with eigen values and eigen vectors of the input covariance. However, since we have now computed the U and S matrix, and know the original input A matrix, we can calculate U matrix using the original SVD relationship:
```math
A = USV^T
```
Since V matrix is orthogonal, inverse(V) is the same as transpose(V), therefore:
```math
AV = US
```
And since S is a diagonal matrix, inverse(S) is the same as doing diagonal-element-wise reciprocals.

So to sum up, Matrix U can be obtained through:
```math
U = AVS^{-1} =
\begin{bmatrix}
(\sum_{k=1}^{n} a_{0,k}v_{k,0})/s_{0,0} & (\sum_{k=1}^{n} a_{0,k}v_{k,1})/s_{1,1} & (\sum_{k=1}^{n} a_{0,k}v_{k,2})/s_{2,2} & ... & (\sum_{k=1}^{n} a_{0,k}v_{k,n})/s_{n,n} & ... \\
(\sum_{k=1}^{n} a_{1,k}v_{k,0})/s_{0,0} & (\sum_{k=1}^{n} a_{1,k}v_{k,1})/s_{1,1} & (\sum_{k=1}^{n} a_{1,k}v_{k,2})/s_{2,2} & ... & (\sum_{k=1}^{n} a_{1,k}v_{k,n})/s_{n,n} & ... \\
(\sum_{k=1}^{n} a_{2,k}v_{k,0})/s_{0,0} & (\sum_{k=1}^{n} a_{2,k}v_{k,1})/s_{1,1} & (\sum_{k=1}^{n} a_{2,k}v_{k,2})/s_{2,2} & ... & (\sum_{k=1}^{n} a_{2,k}v_{k,n})/s_{n,n} & ... \\
...                                     & ...                                     & ...                                     & ... & ...                                     & ... \\
(\sum_{k=1}^{n} a_{m,k}v_{k,0})/s_{0,0} & (\sum_{k=1}^{n} a_{m,k}v_{k,1})/s_{1,1} & (\sum_{k=1}^{n} a_{n,k}v_{k,2})/s_{2,2} & ... & (\sum_{k=1}^{n} a_{m,k}v_{k,n})/s_{n,n} & ... 
\end{bmatrix} 
```
However, not all columns of the U matrix can be constructed this way. When the input matrix is not square, the number of eigen vectors calculated is less than the number of columns in U matrix.

```math 
U = AVS^{-1} =
\begin{bmatrix}
{\sum_{k=1}^{n}} (a_{0,k}v_{k,0})
\end{bmatrix} 
```

In this kernel, filler data is inserted to complete the matrix if needed. 

### U Matrix orthogonalization
As mentioned above, when filler data is inserted to complete the U matrix, it breaks the orthogonality requirement. To fix this, we need to generate column vectors that are orthogonal with the rest of the matrix. 

An efficient algorithm to do this is already implemented in our [QR Decomposition sample](../qrd/README.md). So here we will just insert an instant of the streamingQRD design. The Q output of this kernel is orthogonalized U matrix.

### Demo testbench
In this sample, a testbench is used to demo the SVD design. The test bench use an input matrix of size 16 x 8, and its know singular values to compare for correctness.

Since the singular vectors are non-unique, their correctness are checked by checking its orthogonality and the relationship: 
```math
A = USV^-1
```

## Build the `SVD` design
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
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
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
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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
      The report resides at `pca_report/reports/report.html`.

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
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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
      The report resides at `pca_report.a.prj/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your 'build' directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory, for example:
>
>  ```
  > C:\samples\build> cmake -G "NMake Makefiles" C:\long\path\to\code\sample\CMakeLists.txt
>  ```
## Run the `SVD` Design

### Configurable Parameters

| Argument  | Description
|:---       |:---
| `<num>`   | (Optional) Specifies the number of times to repeat the decomposition. Its default value is **1** for the emulation and simulation flow and **16384** for the FPGA flow.

### On Linux

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./svd.fpga_emu
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./svd.fpga_sim
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./svd.fpga
   ```

### On Windows

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   svd.fpga_emu.exe
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   svd.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   svd.fpga.exe
   ```

## Example Output

Example Output when running on the **Terasic DE10-Agilex Development Board**.

```
Running on device: Intel(R) FPGA Emulation Device
Running SVD test with input size 16 x 8, repeating 16384 time(s)
Using device allocations
Singular value error: 5.96046e-07(9.04024e-05%)
Decomposition error (A = USVt): 9.83477e-07(0.00153999%)
U orthogonal error: 3.05474e-07
V orthogonal error: 9.53674e-07
Total duration: 6.43593s
Throughput: 2.54571k matrices/s
PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).