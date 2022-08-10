# `Monte Carlo Pi Estimation` Sample

Monte Carlo Pi Estimation shows how to use the Intel® oneAPI Math Kernel Library (oneMKL) random number generation (RNG) functionality to estimate the value of &pi;.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows* 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use the oneMKL random number generation functionality
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose
Monte Carlo Pi Estimation uses a randomized method to estimate the value of &pi;. The basic idea is to generate random points within a square, and to check what fraction of these random points lie in a quarter-circle inscribed in that square. The expected value is the ratio of the areas of the quarter-circle and the square, namely &pi;/4, so we can take the observed fraction of points in the quarter-circle as an estimate of &pi;/4.

This sample uses the oneMKL random number generation functionality to produce the random points. oneMKL RNG has APIs that can be called from the host, and APIs that can be called from within a kernel; both kinds of APIs are illustrated.

This sample performs its computations on the default SYCL* device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a SYCL*-compliant buffer or USM pointer, or directly in a kernel.

In this sample, a Philox 4x32x10 generator is used, and a uniform distribution is the basis for the Monte Carlo simulation. oneMKL provides many other generators and distributions to suit a range of applications.

## Using Visual Studio Code* (Optional)
You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the [Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## Building the Monte Carlo Pi Estimation Sample
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
If running a sample in the Intel® DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/).


### On a Linux* System
Run `make` to build and run the sample. Three programs are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

> **Warning**: On Windows, static linking with oneMKL currently takes a very long time, due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Monte Carlo Pi Estimation Sample
### Example of Output
If everything is working correctly, after building you will see step-by-step output from each of the three example programs, providing the generated estimate of &pi;.
```
./mc_pi

Monte Carlo pi Calculation Simulation
Buffer API
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14139
Exact value of Pi = 3.14159
Absolute error = 0.000207487

./mc_pi_usm

Monte Carlo pi Calculation Simulation
Unified Shared Memory API
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14139
Exact value of Pi = 3.14159
Absolute error = 0.000207487

./mc_pi_device_api

Monte Carlo pi Calculation Simulation
Device API
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14139
Exact value of Pi = 3.14159
Absolute error = 0.000207487
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).