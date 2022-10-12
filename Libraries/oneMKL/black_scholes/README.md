# `Black-Scholes` Sample

Black-Scholes shows how to use Vector Math (VM) and Random Number Generator (RNG) functionality available in Intel® oneAPI Math Kernel Library (oneMKL) to calculate the prices of options using the Black-Scholes formula for suitable randomly-generated portfolios.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use the vector math and random number generation functionality in oneMKL
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

The Black-Scholes formula is widely used in financial markets as a basic prognostic tool. The ability to calculate it quickly for a large number of options has become a necessity and represents a classic problem in parallel computation.

The sample first generates a portfolio within given constraints using a uniform distribution and a Philox-type generator provided by the oneMKL RNG API.
Examples of both host-based APIs and device-based APIs for random number generation are provided.

Then, the Black-Scholes formula is used in two distinct implementations:
* scalar kernel containing SYCL scalar functions, which will be parallelized by the Intel® oneAPI DPC++ Compiler
* vector-based implementation, which uses oneMKL vector math functionality in sequential calls to evaluate the transcendental functions used in the formula.

This sample performs its computations on the default SYCL* device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to use oneMKL VM to implement explicit vectorization of calculations over vectors of large size. The key point illustrated is the ability to asynchronously mix kernels with VM function calls and their results. When using USM, the user is in charge of noting explicit dependencies between calls; this essential technique is also illustrated.

This sample also illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a SYCL buffer or USM pointer, or directly in a kernel using device API.

In this sample, a Philox 4x32x10 generator is used. It is a lightweight counter-based RNG well-suited for parallel computing.


## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

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
If running a sample in the Intel® DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### On a Linux* System
Run `make` to build and run the sample. One program is generated, which will use four different interfaces in turn.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Black-Scholes Sample
If everything is working correctly, the program will exercise different combinations of APIs for both single and double precision (if available) and print a summary of its computations.

```
running on:
       device name: Intel(R) Gen9
    driver version: 0.91

running floating-point type float


running USM mkl::rng
    <s0> = 29.9999
    <x> = 29.9994
    <t> = 1.50002
running USM mkl::vm
    <opt_call> = 27.148
    <opt_put>  = 4.12405
```
...
```
running Buffer mkl::rng_device
    <s0> = 30.2216
    <x> = 30.0563
    <t> = 1.49756
running Buffer dpcpp
    <opt_call> = 27.3496
    <opt_put>  = 4.10354
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).