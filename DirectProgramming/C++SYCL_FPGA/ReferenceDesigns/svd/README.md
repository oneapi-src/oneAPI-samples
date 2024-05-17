# `Singular Value Decomposition (SVD)` Sample
The `SVD` reference design demonstrates a singular value decomposition implementation for real matrices on an FPGA.


| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement an FPGA version of singular value decomposition.
| Time to complete      | 1 hr (not including compile time)
| Category              | Reference Designs and End to End

## Purpose
This FPGA reference design demonstrates the Singular Value Decomposition (SVD) of real matrices. SVD is a common linear algebra factorization used for applications such as matrix pseudo-inverse and solving homogeneous linear equations. The SVD of a real, 2D input matrix $A$ is defined as follows:

```math
A = USV^T
```

where $U$ and $V$ are orthonormal bases of $A$, called (left and right) singular vectors. Orthonormal bases are vectors that are orthogonal to each other and of length 1.

$S$ is a diagonal matrix (matrix where all elements but the diagonals are zeros). The diagonals are called singular values corresponding to each singular vectors.

While you can apply SVD to complex matrices, this design does not support complex matrices to keep the design simple.

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

Performance results are based on testing as of April 12, 2024 with fixed 55 iterations.

Using BSP: Agilex_ref 23.3
> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Terasic DE10-Agilex Development Board             | 0.92k matrices/s for matrices of size 32 x 32

## Key Implementation Details
This SVD design consists of 4 computation kernels, as well as several memory access kernels to handle input and output. These kernels are connected through inter-kernel pipes and input/output through unified shared memory (USM).

![](assets/SVD.svg)

### Source file structure
| File name              | Type                            | Content                                                                                                                                                                                              |
| ---------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| svd_demo.cpp           | Host code                       | Launch a demonstration using the SVD design. Contains the `main()` function.                                                                                                                         |
| svd_testcase.hpp       | Host code                       | A `struct` that lunches the design with a set of input and check result for correctness. The input can either be specified or generated.                                                             |
| svd_testbench_tool.hpp  | Host code                       | Helper functions that is used in the test bench of the SVD demonstration to check for correctness.                                                                                                   |
| print_matrix.hpp       | Host code                       | Helper functions to print matrices.                                                                                                                                                                  |
| golden_pca.hpp         | Host code                       | A CPU reference PCA design that is used to calculate reference eigen values for the test bench.                                                                                                      |
| svd.hpp                | Host code with some device code | Contains wrapper function `SingularValueDecomposition` that launches individual kernels of the SVD design.                                                                                           |
| memory_transfers.hpp   | Device code                     | Contains kernel implementations of `MatrixReadFromDDRTo2PipesByBlocks` , `MatrixReadPipeToDDR` and `VectorReadPipeToDDR`. These kernels transfers data between DDR and streaming interfaces (pipes).|
| non_std_covariance.hpp | Device code                     | Contains kernel implementation of `StreamingNStdCovarianceMatrix` that computes covariance matrices.                                                                                                 |
| post_process.hpp       | Device code                     | Contains kernel implementation of `USVFromEigens`.                                                                                                                                                   |
### Input covariance matrix computation
The covariance computation in this design is the same as used in the [PCA](../pca/README.md) reference design except without standardization.

The covariance of input A is equal to the transpose of A multiplied by A:
```math
Cov(A) = A^{T}A
```
Therefore this kernel performs a matrix multiplication by blocks, as described in the PCA reference design.

### Eigenvalues and Eigenvectors computation
We are reusing the `fpga_linalg::StreamingEigen` design from the PCA sample to compute eigenvalues and eigenvectors of the input covariance matrix. These eigenvalues and eigenvectors are going to be used to construct the final outputs of SVD.

This kernel works only with input of rank sufficient matrix (all columns are linearly independent). It produces an output `rank_deficent_flag` to indicate if the input matrix is not linearly independent. If the flag is set to 1, the result of the SVD is known to be incorrect.

The source code for the the eigenvalues and eigenvectors computation kernel can be found in `streaming_eigen.hpp` in the shared [include](../../include) directory. An detailed explanation of the algorithm of computing eigenvalues can be found [here](../pca/README.md#eigen-values-and-eigen-vectors-computation) in the PCA sample.

### Construct Output from Eigenvalues and Eigenvectors
The final outputs of SVD can be constructed from the eigenvalues and eigenvectors of the input covariance. 

Consider an input matrix $A$ of size $m \times n$.

#### $V$ Matrix
The right singular vectors (the $V$ matrix) can be produced by copying the eigenvectors , and should be of size $n \times n$:
```math
V = [v_0, v_1, v_2, ...,v_n] 
```

#### $S$ Matrix
The singular value matrix (the $S$ matrix) can be constructed as an $m \times n$ diagonal matrix (all but the values on the main diagonal are zeros) where the non-zero values are the square root of each eigenvalue:
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

#### $U$ Matrix
The left singular vectors (the $U$ matrix) cannot be directly constructed with eigenvalues and eigenvectors of the input covariance. However, since we have now computed the $U$ and $S$ matrix, and knowing the original input $A$ matrix, we can calculate $U$ matrix using the original SVD relationship:
```math
A = USV^T
```
Since the $V$ matrix is orthogonal by construction, $V^{-1}$ is the same as $V^T$, therefore:
```math
AV = US
```
And since $S$ is a diagonal matrix, $S^{-1}$ is the same as multiplying by the reciprocals of each element on the main diagonal of $S$.

In summary, the $m \times n$ portion of matrix $U$ ($U_{[0:m][0:n]}$) can be obtained through:
![](assets/U_matrix.png)

When the input matrix $A$ is not square, the number of eigenvectors calculated is less than the number of columns in the $U$ matrix. An orthogonalization kernel is needed to generate extra orthogonal vectors.

### $U$ Matrix Orthogonalization
As mentioned above, when extra filler vectors are needed to complete the $U$ matrix, we need to orthogonalize the matrix.

An efficient algorithm to do this is already implemented in our [QR Decomposition sample](../qrd/README.md), so here we will insert an instance of the `fpga_linalg::streamingQRD` design. The $Q$ output of this kernel is orthogonalized $U$ matrix. Since we only care about the orthogonalized $U$ matrix, $R$ output of the Streaming QRD kernel is discarded.

### Demonstration Testbench
In this sample, a testbench is used to demonstrate the SVD design.

The resulting singular values are checked against eigen values calculated by a reference PCA algorithm that is also used in the PCA sample.

Since the singular vectors in $U$ and $V$ are non-unique, their correctness are checked by (1) checking for orthogonality, and (2) that they satisfy the relationship $A = USV^T$.

## Build the `SVD` Design
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
      The report resides at `svd_report.prj/reports/report.html`.

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
      The report resides at `svd_report.a.prj/reports/report.html`.

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

## Example Output

Example Output when running on the **Terasic DE10-Agilex Development Board**.

```
Running on device: de10_agilex : Agilex Reference Platform (aclde10_agilex0)
Running SVD test with input size 16 x 8, repeating 16384 time(s)
Using device allocations
Singular value error: 5.96046e-07(9.04024e-05%)
Decomposition error (A = USVt): 9.83477e-07(0.00153999%)
U orthogonal error: 3.05474e-07
V orthogonal error: 9.53674e-07
Total duration: 0.989917s
Throughput: 16.5509k matrices/s
PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).