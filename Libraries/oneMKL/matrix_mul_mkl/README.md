# `Matrix Multiplication with oneMKL` Sample

The `Matrix Multiplication with oneMKL` sample demonstrates how to use Intel® oneAPI Math Kernel Library (oneMKL) optimized matrix multiplication routines.

| Area                | Description
|:---                 |:---
| What you will learn | How to use the oneMKL matrix multiplication functionality.
| Time to complete    | 15 minutes

>**Note**: For more information and complete documentation of all oneMKL routines, see the [Intel® Math Kernel Library Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html).

## Purpose

This sample uses oneMKL to multiply two large matrices. The sample code performs computations on the default SYCL* device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                  | Ubuntu* 20.04, 22.04 <br> Windows* 10, 11
| Hardware            | Intel Atom® Processors <br> Intel® Core™ Processor Family <br> Intel® Xeon® Processor Family 
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)

## Key Implementation Details

The oneMKL `blas::gemm` routine performs a generalized matrix multiplication operation. oneMKL BLAS routines support both row-major and column-major matrix layouts; this sample uses row-major layouts, the traditional choice for C++.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build and Run the `Matrix Multiplication with oneMKL` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the *[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

### On Linux*

1. Change to the sample directory.
2. Build and run the program.
   ```
   cmake ..
   make
   ```
3. Remove generated files. (Optional)
   ```
   make clean
   ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Intel® Agilex® device family, which is the default.
   ```
   cmake -G "NMake Makefiles" ..
   ```
   >**Warning**: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

3. Remove generated files. (Optional)
   ```
   nmake clean
   ```
   > **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

### Build and Run the Sample in Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.

Use the Linux instructions to build and run the program.

If needed, you can specify a CPU node using a single line script.

```
qsub -l nodes=1:xeon:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:xeon:ppn=2` (lower case L) assigns one full GPU node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes  |Command Options
  |:---             |:---
  |GPU	            |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	            |`qsub -l nodes=1:xeon:ppn=2 -d .`
  
  >**Note**: For more information on how to specify compute nodes read *[Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/)* in the Intel® DevCloud Documentation.

For more information on using Intel® DevCloud, see the Intel® oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started/) page.

## Example Output

The program generates two input matrices and call oneMKL to multiply them. It will also compute the product matrix itself to verify the results from oneMKL.

```
./matrix_mul_mkl
Device: Intel(R) Gen9 HD Graphics NEO
Problem size:  A (600x1200) * B (1200x2400)  -->  C (600x2400)
Launching oneMKL GEMM calculation...
Performing reference calculation...
Results are accurate.

./matrix_mul_mkl_bf16
C = [ 0, 0, ... ]
    [ 0, 0,  ... ]
    [ ... ]
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).