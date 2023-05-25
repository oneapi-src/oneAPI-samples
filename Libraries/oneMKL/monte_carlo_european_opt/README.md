# `Monte Carlo European Options` Sample

Monte Carlo European Options shows how to use the Intel® oneAPI Math Kernel Library (oneMKL)
random number generation (RNG) functionality to compute European option prices.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows* 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use the oneMKL random number generation functionality
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines,
see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

This sample uses a Monte Carlo (randomized) method to estimate European call and put option values
based on a stochastic stock price model. A large number of possible implementations of the stock
price over time are generated, and an estimation of the call and put options is made by averaging
their values in each realization.

This sample performs its computations on the default SYCL* device. You can set
the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.

## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness),
a distribution object (specifying the desired probability distribution), and finally generate
the random numbers themselves. Random number generation can be done from the host,
storing the results in a SYCL-compliant buffer or USM pointer, or directly in a kernel.

In this sample, the MCG59 generator is used by default, and a gaussian distribution
is the basis for the Monte Carlo simulation. oneMKL provides many other generators
and distributions to suit a range of applications. After generating the random number
input for the simulation, prices are calculated and then averaged using reduction functions.

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

## Building the Monte Carlo European Options Sample
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

> **Warning**: On Windows, static linking with oneMKL currently takes a very long time, due to a known compiler issue. This will be addressed in an upcoming release.

#### Build a sample using others generators
To use the MRG32k3a generator or the Philox4x32x10 generator use `generator=mrg`
or `generator=philox` correspondingly when building the sample, e.g.
```
make generator=mrg
```
for Linux* system or

```
nmake generator=mrg
```

for Windows* System.

#### Build a sample using Random Number Generation on Host
To have random number generation on host to initialize data use
`init_on_host=1`, e.g.
```
make init_on_host=1
```
for Linux* system or

```
nmake init_on_host=1
```

for Windows* System.

## Running the Monte Carlo European Options Sample
If everything is working correctly, the program will run the Monte Carlo simulation.
After the simulation, results will be checked against the known true values
given by the Black-Scholes formula, and the absolute error is output.

Example of output:
```
$ ./montecarlo

MonteCarlo European Option Pricing in Double precision
Pricing 384000 Options with Path Length = 262144, sycl::vec size = 8, Options Per Work Item = 4 and Iterations = 5
Completed in 67.6374 seconds. Options per second = 22709.3
Running quality test...
L1_Norm          = 0.000480579
Average RESERVE  = 12.9099
Max Error        = 0.123343
TEST PASSED!

```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
