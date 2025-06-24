# `Multiple Simple Random Sampling without replacement` Sample

Multiple Simple Random Sampling without replacement shows how to use the Intel® oneAPI Math Kernel Library (oneMKL) random number generation (RNG) functionality to generate K>>1 simple random length-M samples without replacement from a population of size N (1 ≤ M ≤ N).

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel® oneAPI Math Kernel Library (oneMKL)
| What you will learn | How to use the oneMKL random number generation functionality
| Time to complete    | 15 minutes

For more information on oneMKL and complete documentation of all oneMKL routines, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html.

## Purpose

The sample demonstrates the Partial Fisher-Yates Shuffle algorithm conducts 11 969 664 experiments. Each experiment, which generates a sequence of M unique random natural numbers from 1 to N, is a partial length-M random shuffle of N elements' whole population. Because the algorithm's main loop works as a real lottery, each experiment is called "lottery M of N" in the program.
The program uses M=6 and N=49 and stores result samples (sequences of length M) in a single array.

This sample uses the oneMKL random number generation functionality to produce random numbers. oneMKL RNG has APIs that can be called from the host, and APIs can be reached from within a kernel; both kinds of APIs are illustrated.

This sample performs its computations on the default SYCL* device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a SYCL-compliant buffer or USM pointer or directly in a kernel.

In this sample, a Philox 4x32x10 generator is used, and a uniform distribution is a basis for the algorithm. oneMKL provides many other generators and distributions to suit a range of applications.

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## Building the Multiple Simple Random Sampling without replacement Sample
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
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On a Linux* System
Run `make` to build and run the sample. Three programs are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `make clean.`

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time due to a known compiler issue. This will be addressed in an upcoming release.

## Running the Multiple Simple Random Sampling without replacement Sample

### Example of Output
After building, if everything is working correctly, you will see the step-by-step output from each of the three example programs, providing the lottery results.
```
./lottery

Multiple Simple Random Sampling without replacement
Buffer Api
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,

TEST PASSED

./lottery_usm

Multiple Simple Random Sampling without replacement
Unified Shared Memory API
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Results with Host API:
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,

TEST PASSED

./lottery_device_api

Multiple Simple Random Sampling without replacement
Device API
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,

TEST PASSED
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
