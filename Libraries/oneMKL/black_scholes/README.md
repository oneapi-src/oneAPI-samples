# `Black-Scholes` Sample

Black-Scholes shows how to use Random Number Generator (RNG) functionality
available in Intel® oneAPI Math Kernel Library (oneMKL) to calculate the prices
of options using the Black-Scholes formula for suitable randomly-generated portfolios.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows 10
| Hardware            | Skylake CPU with Gen9 GPU or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use the oneMKL random number generation functionality
| Time to complete    | 5 minutes

For more information on oneMKL and complete documentation of all oneMKL routines,
see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

The Black-Scholes formula is widely used in financial markets as a basic
prognostic tool. The ability to calculate it quickly for a large number of
options has become a necessity and represents a classic problem in parallel
computation.

The sample first generates a portfolio within given constraints using a uniform
distribution and a Philox-type generator provided by the oneMKL RNG API.

This sample performs its computations on the default SYCL* device. You can set
the `SYCL_DEVICE_FILTER` environment variable to `cpu` or `gpu` to select the device to use.

## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of
pseudo-randomness), a distribution object (specifying the desired probability
distribution), and finally generate the random numbers themselves.

In this sample, a Philox 4x32x10 generator is used. It is a lightweight
counter-based RNG well-suited for parallel computing.

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## Building the Black-Scholes Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Running Samples In Intel® DevCloud
If running a sample in the Intel® DevCloud, remember that you must specify the
compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode.
For more information see the Intel® oneAPI Base Toolkit Get Started Guide
(https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### On a Linux* System
Run `make` to build the sample. Then run the sample calling generated execution file.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample programs. `nmake clean` removes temporary files.

#### Build a sample using Random Number Generation on Host
To use the RNG on host use `init_on_host=1`, e.g.
```
make init_on_host=1
```
for Linux* system or

```
nmake init_on_host=1
```

for Windows* System.

## Running the Black-Scholes Sample
If everything is working correctly, the program will run the Black-scholes simulation.
After the simulation, results will be checked against the known true values
given by the Black-Scholes formula, and the absolute error is output.

Example of output:
```
$ ./black_scholes_sycl

Double Precision Black&Scholes Option Pricing version 1.6 running on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz using DPC++, workgroup size 256, sub-group size 32.
Compiler Version: Intel(R) oneAPI DPC++/C++ Compiler 2023.1.0 (2023.1.0.20230320), LLVM 16.0 based.
Driver Version  : 2023.15.3.0.20_160000
Build Time      : Apr 28 2023 05:34:33
Input Dataset   : 8388608
Pricing 16777216 Options in 512 iterations, 8589934592 Options in total.
Completed in    1.41111 seconds. GOptions per second:    6.08735
Time Elapsed =     1.41111 seconds
Creating the reference result...
L1 norm: 1.385136E-16
TEST PASSED

```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
