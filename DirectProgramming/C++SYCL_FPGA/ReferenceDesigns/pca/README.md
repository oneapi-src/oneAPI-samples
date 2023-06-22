# `Principal Component Analysis (PCA)` Sample
The `PCA` reference design demonstrates a principal component analysis implementation for real matrices on an FPGA.


| Area                  | Description
|:---                   |:---
| What you will learn   | How to implement an FPGA version of principal component analysis.
| Time to complete      | 1 hr (not including compile time)
| Category              | Reference Designs and End to End

## Purpose

This FPGA reference design demonstrates the Principal Component Analysis (PCA) of real matrices, a common linear algebra operation used to analyze large datasets in machine learning applications. 

Real-world datasets typically consist of multiple features $Nf$ describing a large number of samples $Ns$.
However, not all features contain significant information, and some may be nearly identical across the dataset.
In order to facilitate analysis, visualization, and storage of the data, reducing the number of features while maintaining the majority of the information is crucial.
The PCA approach identifies such principal features, in descending order based on their contribution to the information within the samples.
These principal features are identified by the weight of their associated Eigen Values.
The result of the PCA of an input Matrix *A* is then defined as its sorted Eigen values and Eigen Vectors.

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

Performance results are based on testing as of June 1st, 2023.

> **Note**: Refer to the [Performance Disclaimers](/DirectProgramming/C++SYCL_FPGA/README.md#performance-disclaimers) section for important performance information.

| Device                                            | Throughput
|:---                                               |:---
| Intel® PAC with Intel® Arria® 10 GX FPGA          | 7.7k matrices/s for matrices of size 8 features * 4176 samples
| Terasic DE10-Agilex Development Board             | 16k matrices/s for matrices of size 8 features * 4176 samples

## Key Implementation Details

The algorithm is split in two parts:
1/ the input matrix is standardized and transformed to a covariance matrix
2/ the Eigen values and Eigen vectors are computed using the QR iteration process

### Input matrix standardization and covariance matrix computation

The standardized covariance matrix is defined as:
```math
Cov[i][j] = \frac{1}{N-1}*\sum_{k=0}^{N-1}{(A_{std}[k][i] - mean_{std}[i])(A_{std}[k][j] - mean_{std}[j]) }
```

Where $A_{std}$ is the standardized matrix $A$, defined as:
```math
A_{std}[i][j] = \frac{A[i][j] - mean[j]}{s_j}
```
with $s_j$ being the standard deviation of the column $j$:
```math
s_j = \frac{1}{\sqrt{N-1}}*\sqrt{\sum_{k=0}^{N-1}(A[k][j] - mean[j])^2}
```
and $mean_{std}[j]$ being the mean of the column $j$ of the standardized A matrix.

By rewriting these equations, we can obtain something that will better map to an FPGA: 
```math
Cov[i][j] = \frac{T[i][j] - N*mean[i]*mean[j]}{\sqrt{(T[i][i] -N*mean[i]^2) * (T[j][j] -N*mean[j]^2)}}
```

where $T$ is defined as the transposed input matrix multiplied by the input matrix: 
```math
T[i][j] = \sum_{k=0}^{N-1}(A[k][i]*A[k][j])
```

Using these equations, we can write a function that implements the matrix product $T$, which also computes the mean of each column of $A$ while doing so.

The number of samples in the $A$ matrix $Ns$ is typically very large, resulting in a very tall matrix $A$ matrix.
Implementing the computation of $T$ using a single dot-product of size $Ns$ would then result in a very large DSP usage, and potentially not even fit in the target FPGA.

As an example, let's consider this $Ns = 8, Nf = 4$ example:

```math
\begin{bmatrix}
A_{0,0} & A_{1,0} & A_{2,0} & A_{3,0} & A_{4,0} & A_{5,0} & A_{6,0} & A_{7,0} \\
A_{0,1} & A_{1,1} & A_{2,1} & A_{3,1} & A_{4,1} & A_{5,1} & A_{6,1} & A_{7,1} \\
A_{0,2} & A_{1,2} & A_{2,2} & A_{3,2} & A_{4,2} & A_{5,2} & A_{6,2} & A_{7,2} \\
A_{0,3} & A_{1,3} & A_{2,3} & A_{3,3} & A_{4,3} & A_{5,3} & A_{6,3} & A_{7,3} 
\end{bmatrix} 
\times 
\begin{bmatrix}
A_{0,0} & A_{0,1} & A_{0,2} & A_{0,3} \\
A_{1,0} & A_{1,1} & A_{1,2} & A_{1,3} \\
A_{2,0} & A_{2,1} & A_{2,2} & A_{2,3} \\
A_{3,0} & A_{3,1} & A_{3,2} & A_{3,3} \\
A_{4,0} & A_{4,1} & A_{4,2} & A_{4,3} \\
A_{5,0} & A_{5,1} & A_{5,2} & A_{5,3} \\
A_{6,0} & A_{6,1} & A_{6,2} & A_{6,3} \\
A_{7,0} & A_{7,1} & A_{7,2} & A_{7,3} 
\end{bmatrix} 
= 
\begin{bmatrix}
T_{0,0} & T_{0,1} & T_{0,2} & T_{0,3} \\
T_{1,0} & T_{1,1} & T_{1,2} & T_{1,3} \\
T_{2,0} & T_{2,1} & T_{2,2} & T_{2,3} \\
T_{3,0} & T_{3,1} & T_{3,2} & T_{3,3} \\
\end{bmatrix}
```

To compute any element of the $T$ matrix in a single iteration, one would need a dot-product engine of size 8.
The dataset used in this reference design has 4176 samples, which would require a dot-product engine of size 4176.
As this method does not scale to matrices with a large number of samples, we use a blocked matrix product approach.

At each block iteration, a $Nf \times Nf$ submatrix of $A$ is read and used to compute the partial matrix product $Tp$.
Illustration of this mechanism after block $0$ has been computed:
```math
\begin{bmatrix}
A_{0,0} & A_{1,0} & A_{2,0} & A_{3,0} & - & - & - & - \\
A_{0,1} & A_{1,1} & A_{2,1} & A_{3,1} & - & - & - & - \\
A_{0,2} & A_{1,2} & A_{2,2} & A_{3,2} & - & - & - & - \\
A_{0,3} & A_{1,3} & A_{2,3} & A_{3,3} & - & - & - & -
\end{bmatrix} 
\times 
\begin{bmatrix}
A_{0,0} & A_{0,1} & A_{0,2} & A_{0,3} \\
A_{1,0} & A_{1,1} & A_{1,2} & A_{1,3} \\
A_{2,0} & A_{2,1} & A_{2,2} & A_{2,3} \\
A_{3,0} & A_{3,1} & A_{3,2} & A_{3,3} \\
- & - & - & - \\
- & - & - & - \\
- & - & - & - \\
- & - & - & - 
\end{bmatrix} 
+ 
\begin{bmatrix}
Tp_{0,0} & Tp_{0,1} & Tp_{0,2} & Tp_{0,3} \\
Tp_{1,0} & Tp_{1,1} & Tp_{1,2} & Tp_{1,3} \\
Tp_{2,0} & Tp_{2,1} & Tp_{2,2} & Tp_{2,3} \\
Tp_{3,0} & Tp_{3,1} & Tp_{3,2} & Tp_{3,3}
\end{bmatrix}
```

After enough iterations, the $Tp$ accumulated values will hold the result of the entire matrix product.
Partial means are also computed alongside the matrix multiplication.
Then, the covariance matrix can simply be computed the covariance equation above.

### Eigen values and Eigen vectors computation

The Eigen values and Eigen vectors are computed using the QR iteration process.
The algorithm iterates until the Eigen values have been found.

**Set** $C=Cov$ <br /> 
**do** <br /> 
   &emsp; **QR Decomposition** $[Q,R] = qrd(C)$ <br /> 
   &emsp; Set $C=R \times Q$ <br /> 
**while** ($C$ is not a diagonal matrix)
<br />

Upon achieving convergence, the diagonal values of $C$ will contain the Eigen values. 
Although this approach is mathematically correct, the number of iterations required to reach convergence is very high.
To reduce the number of required iterations, one common technique is to apply the _shifted_ QR iteration process.
At each iteration, the next Eigen value is first approximated using the _Rayleigh_ shift method.
This shift value is subtracted from the diagonal of $C$ before the QR decomposition is performed:

**Set** $C=Cov$ <br /> 
**do** <br /> 
   &emsp; **Compute Rayleigh shift**: $\mu = shift(C)$ <br /> 
   &emsp; **Subtract the shift from the diagonal elements** $C = C - (Id \times \mu)$ <br /> 
   &emsp; **QR Decomposition**: $[Q,R] = qrd(C)$ <br /> 
   &emsp; Set $C=(R \times Q) + (Id \times \mu)$ <br /> 
**while** ($C$ is not a diagonal matrix)
<br />

The behavior of this algorithm is that the last row and column of $C$ will converge towards containing only zeroes, except the diagonal element containing the Eigen value.
Once the last row has converged, we can focus on the convergence of the submatrix of $C$ with the last row/column removed.
Because the algorithm gradually computes the Eigen values, the associated shift value must be computed based on different values from the $C$ matrix depending on the current Eigen value to find.
The Wilkinson shift $\mu$ computation requires bottom right $2 \times 2$ elements of the current sub-matrix of $C$ to compute the shift value:

```math
\begin{bmatrix}
x & x & x & x \\
x & x & x & x \\
x & x & a & b \\
x & x & b & c \\
\end{bmatrix}
``` 

The Wilkinson shift is then given by following equation:

```math
\mu = c - \frac{sign(\delta) \times b^{2}}{|\delta| + \sqrt{\delta^{2} + b^{2}}}
```
where $\delta = (a − c)/2$.

The Eigen vectors ($E_{vec}$) can be computed by compounding the $Q$ matrix computed from the QR decomposition in each QR iteration as follows:

**Set** $C=Cov$ <br /> 
**Set** $E_{vec}=Id$ <br />
**do** <br /> 
   &emsp; **Compute Rayleigh shift**: $\mu = shift(C)$ <br /> 
   &emsp; **Subtract the shift from the diagonal elements** $C = C - (Id \times \mu)$ <br /> 
   &emsp; **QR Decomposition**: $[Q,R] = qrd(C)$ <br /> 
   &emsp; **Update the Eigen vectors**: $E_{vec} = E_{vec} \times Q$ <br /> 
   &emsp; Set $C=(R \times Q) + (Id \times \mu)$ <br /> 
**while** ($C$ is not a diagonal matrix)
<br />

After the QR iteration is complete, the Eigen values and Eigen vectors are sorted by significance.
The sorting actually creates an index list of the most significant columns.
This index is then used to stream the Eigen values and Eigen vectors out of the kernel in the correct order.


More information about the QR decomposition algorithm used in this design can be found in the [QRD reference design](/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/qrd).


To optimize the performance-critical loop in this algorithm, the design leverages concepts discussed in the following FPGA tutorials:

- **Triangular Loop Optimization** (triangular_loop)
- **Explicit Pipelining with `fpga_reg`** (fpga_register)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Unrolling Loops** (loop_unroll)
- **Memory attributes** (memory_attributes)
- **Explicit data movement** (explicit_data_movement)
- **Algorithmic C Integer Data Type** (ac_int)
- **Loop initiation interval** (loop_initiation_interval)
- **Data Transfers Using Pipes** (pipes)

The key optimization techniques used are as follows:
1. Rewriting the equation of the standardized covariance matrix computation to better map FPGA hardware
2. Implement the $T$ matrix computation in a blocked fashion
3. Refactoring the original Gram-Schmidt algorithm to merge two dot products into one, reducing the total number of dot products needed to three from two. This helps us reduce the DSPs required for the implementation.
4. Converting the nested loop into a single merged loop and applying Triangular Loop optimizations. This allows us to generate a design that is very well pipelined.
5. Fully vectorizing the dot products using loop unrolling.
6. Using an efficient memory banking scheme to generate high performance hardware.
7. Using the `fpga_reg` attribute to insert more pipeline stages where needed to improve the frequency achieved by the design.
8. Merging the $R \times Q$ and the $E_{vec} \times Q$ loops to reduce latency.
9. Sorting the Eigen values and Eigen vectors by populating an index list for streaming the columns in the correct order rather than sorting the Eigen values and Eigen vectors in place.

### Dataset used to validate the sample

The dataset used in this sample is the [Abalone dataset](https://archive.ics.uci.edu/ml/datasets/abalone) which is used to predicting the age of abalone from physical measurements.
It can be found in the `data` folder.

### Compiler Flags Used

| Flag                   | Description
|:---                    |:---
| `-Xshardware`          | Target FPGA hardware (as opposed to FPGA emulator)
| `-Xssimulation`        | Target FPGA simulator (as opposed to FPGA emulator)
| `-Xsparallel=2`        | Use 2 cores when compiling the bitstream through Intel® Quartus®
| `-qactypes`            | Link against the `ac_int` libraries 

Additionally, the `cmake` build system can be configured using the following parameters:

| `cmake` option               | Description
|:---                          |:---
| `-DSET_BENCHMARK=[0/1]`      | Specifies if the program is going to be run using it's benchmark mode (`1` by default). In benchmark mode, the path to the dataset file must be passed as a program argument when running the program.
| `-DSET_FEATURES_COUNT=[N]`   | When in non-benchmark mode, set the number of features to `N`.
| `-DSET_SAMPLES_COUNT=[N]`    | When in non-benchmark mode, set the number of samples to `N`.
| `-DSET_FIXED_ITERATIONS=[N]` | Used to set the ivdep safelen attribute for the performance critical triangular loop in the QR decomposition.

>**Note**: The values for `-DSET_FIXED_ITERATIONS` depends on the value of  `-DSET_FEATURES_COUNT`, `-DSET_SAMPLES_COUNT`, the target FPGA and the target clock frequency.

## Build the `PCA` Design

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
>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `C:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `PCA` Design

### Configurable Parameters

| Argument  | Description
|:---       |:---
| `<path>`  | (Required in benchmark mode, must be omitted otherwise) Path to the `abalone.csv` file located in the data folder of this sample.
| `<num>`   | (Optional) Specifies the number of times to repeat the decomposition. Its default value is **1** for the emulation and simulation flow and **4096** for the FPGA flow.

### On Linux

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./pca.fpga_emu <path to data/abalone.csv>
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./pca.fpga_sim <path to data/abalone.csv>
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./pca.fpga <path to data/abalone.csv>
   ```

### On Windows

#### Run on FPGA Emulator

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   pca.fpga_emu.exe <path to data/abalone.csv>
   ```

#### Run on FPGA Simulator

1. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   pca.fpga_sim.exe <path to data/abalone.csv>
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

#### Run on FPGA

1. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   pca.fpga.exe <path to data/abalone.csv>
   ```

## Example Output

Example Output when running on **Intel® PAC with Intel® Arria® 10 GX FPGA**.

```
Running on device: pac_a10 : Intel PAC Platform (pac_f300000)
Reading the input data from file.
Features count: 8
Samples count: 4176
Running Principal Component analysis of 1 matrix 4096 times
Using device allocations
   Total duration:   0.531844 s
Throughput: 7.70151k matrices/s
Verifying results...
All the tests passed.
```

Example Output when running on the **Terasic DE10-Agilex Development Board**.

```
Running on device: de10_agilex : Agilex Reference Platform (aclde10_agilex0)
Reading the input data from file.
Features count: 8
Samples count: 4176
Running Principal Component analysis of 1 matrix 4096 times
Using device allocations
   Total duration:   0.250902 s
Throughput: 16.3251k matrices/s
Verifying results...
All the tests passed.
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
