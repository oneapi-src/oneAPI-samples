# `Matrix Multiply` Sample
The `Matrix Multiply` is a simple program that multiplies together two large matrices and verifies the results. This program uses two methods to accomplish the same operations: SYCL* and OpenMP (OMP).

| Property                   | Description
|:---                        |:---
| What you will learn        | How to offload computations on 2D arrays to a GPU using SYCL* and OpenMP.
| Time to complete           | 15 minutes

> **Note**: This `Matrix Multiply` sample does not support OpenMP on Windows.

### Purpose

The `Matrix Multiply` sample program includes SYCL*-compliant and OpenMP C++ implementations. Each implementation is contained in an appropriately named file: `matrix_mul_dpcpp.cpp` and `matrix_mul_omp.cpp`. The separation provides a way to compare existing offload techniques such as OpenMP with SYCL* within a relatively simple sample. 

The code will attempt to execute on an available GPU first and fallback to the
system CPU if a compatible GPU is not detected. The device used for the
compilation is displayed in the output.

## Prerequisites

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br> Windows 10*
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler  <br> Intel&reg; oneAPI C++ Compiler Classic <br> Intel&reg; C++ Compiler


## Key implementation details

Since the sample multiples two large matrices, this sample is a slightly more complex computation than the [`Vector Add`](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/DenseLinearAlgebra/vector-add) sample. This sample uses buffers to manage memory. (For more information on different memory management options, refer to the `Vector Add` sample.)

The default sample builds the SYCL*-compliant application, and separate OpenMP build instructions are included below.

The code attempts to run the calculation on both the GPU and CPU and verify the results. The size of the computation can be adjusted for heavier workloads. If successful, the name of the offload device and a success message is displayed.

## Build the `Matrix Multiply` Samples

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - For PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files
The include folder is located at `"%ONEAPI_ROOT%\dev-utilities\latest\include"` on your development system. You will need to use some of the resources from this location to build the sample.

### Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux
#### Build for SYCL*
1. Build the program.
   ```
   make all
   ```

#### Build for OpenMP
1. Build the program using Make
   ```
   make build_omp
   ```

### On Windows

#### Build for SYCL*
> **Note**: The OpenMP offload target is not supported on Windows.

1. Build using nmake.
   ```
   nmake -f Makefile.win build_dpcpp
   ```
### On Windows* Using Visual Studio*
- Build the program using VS2017 or later.
    - Right-click on the solution file and open using the IDE.
    - Right-click on the project in **Solution Explorer** and select **Rebuild**.
    - From the top menu, select **Debug** > **Start without Debugging**.

- Build the program using MSBuild.
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio version.
     - Run the following command: `MSBuild matrix_mul.sln /t:Rebuild /p:Configuration="release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors and other issues. See the [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Matrix Multiply` Samples

### On Linux
1. Run both versions.
   ```
   make run
   make run_omp
   ```
### On Windows
1. Run the SYCL version.
   ```
   nmake -f Makefile.win run_dpcpp
   ```
2. Alternatively, change directory to the output folder and run the .exe file. (Depending on the build method you selected, the .exe name could be different from what is shown.)
   ```
   matrix_mul_dpcpp.exe
   ```

### Application Parameters
You can modify the computation size by adjusting the size parameter in the `matrix_mul_dpcpp.cpp` and `matrix_mul_omp.cpp` files. You can configure the following parameters in both files:
```C++
size = m_size = 150*8; // Must be a multiple of 8.
M = m_size / 8;
N = m_size / 4;
P = m_size / 2;
```
> **Note**: The size value must be in multiples of **8**.

### Run the `Matrix Multiply` Sample in Intel&reg; DevCloud
When running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see Intel&reg; oneAPI Base Toolkit [Get Started](https://devcloud.intel.com/oneapi/get_started/).

## Example Output

### SYCL*
```
Running on device: Intel(R) Gen9 HD Graphics NEO
Problem size: c(150,600) = a(150,300) * b(300,600)
Result of matrix multiplication using DPC++: Success - The results are correct!
```

### OpenMP
```
Problem size: c(150,600) = a(150,300) * b(300,600)
Running on 1 device(s)
The default device id: 0
Result of matrix multiplication using OpenMP: Success - The results are correct!
Result of matrix multiplication using GPU offloading: Success - The results are correct!
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).