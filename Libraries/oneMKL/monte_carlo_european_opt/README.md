# `Monte Carlo European Options` Sample

Monte Carlo European Options shows how to use the oneMKL library's random number generation (RNG) functionality to compute European option prices.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL
| What you will learn | How to use oneMKL's random number generation functionality
| Time to complete    | 15 minutes


## Purpose

This sample uses a Monte Carlo (randomized) method to estimate European call and put option values based on a stochastic stock price model. A large number of possible realizations of the stock price over time are generated, and an estimation of the call and put options is made by averaging their values in each realization.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a DPC++ buffer or USM pointer, or directly in a DPC++ kernel.

In this sample, a Philox 4x32x10 generator is used, and a log-normal distribution is the basis for the Monte Carlo simulation. oneMKL provides many other generators and distributions to suit a range of applications. After generating the random number input for the simulation, prices are calculated and then averaged using DPC++ reduction functions.

Two versions of the sample are provided: one using DPC++ buffer objects and one using raw pointers (Unified Shared Memory).


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building the Monte Carlo European Options Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample. Two programs will be generated: a buffer version (`mc_european`) and a USM version (`mc_european_usm`).

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample programs. `nmake clean` removes temporary files.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time, due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Monte Carlo European Options Sample
If everything is working correctly, both buffer and USM versions of the program will run the Monte Carlo simulation. After the simulation, results will be checked against the known true values given by the Black-Scholes formula, and the absolute error is output.

```
./mc_european

Monte Carlo European Option Pricing Simulation
Buffer API
----------------------------------------------
Number of options = 2048
Number of samples = 262144
put_abs_err  = 0.0372205
call_abs_err = 0.0665814

./mc_european_usm

Monte Carlo European Option Pricing Simulation
Unified Shared Memory API
----------------------------------------------
Number of options = 2048
Number of samples = 262144
put_abs_err  = 0.0372205
call_abs_err = 0.0665814
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)