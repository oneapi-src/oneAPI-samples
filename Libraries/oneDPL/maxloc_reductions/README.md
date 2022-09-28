# Maxloc Reduction Sample
Finding the location of the maximum value (maxloc) is a common search operation performed on arrays. It's such a common operation that many programming languages and libraries provide intrinsic maxloc functions (e.g., the C++ max_element function). This sample shows four ways to implement the maxloc reduction using SYCL and oneDPL.

For more information on oneDPL, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-library.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10*
| Hardware            | Intel&reg; Skylake with Gen9 or newer
| Software            | Intel&reg; oneAPI DPC++ Library (oneDPL)
| What you will learn | How to implement the maxloc reduction using oneAPI
| Time to complete    | 15 minutes

## Purpose
This sample demonstrates four ways to implement the maxloc reduction using oneAPI:

 1. maxloc_operator.cpp implements maxloc as a SYCL reduction operator
 2. maxloc_implicit.cpp uses the oneDPL max_element and distance functions and implicit host-device data transfer
 3. maxloc_buffered.cpp uses oneDPL and SYCL buffers for explicit host-device data transfer
 4. maxloc_usm.cpp uses oneDPL and SYCL unified shared memory for host-device data transfer

## Key Implementation Details
[The Maxloc Reduction in oneAPI](https://www.intel.com/content/www/us/en/developer/articles/technical/the-maxloc-reduction-in-oneapi.html) provides more detailed descriptions of each example code, and discusses the relative merits of each approach.

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

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

## Building and Running the Maxloc Reduction Sample

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

### Running Samples on the DevCloud
When running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/).


### On a Linux System
Run `make` to build and run the sample. Two programs are generated: one that uses explicit buffering and one that uses USM.

You can remove all generated files with `make clean`.

### On a Windows System
Run `nmake` to build and run the sample.

Note: To removes temporary files, run `nmake clean`.

*Warning*: On Windows, static linking with oneMKL currently takes a very long time, due to a known compiler issue. This will be addressed in an upcoming oneMKL compiler release.

### Example of Output
If everything is working correctly, each example program will output its target hardware and the location of the maximum value in the input vector:
```
./maxloc_operator
Run on Intel(R) UHD Graphics P630 [0x3e96]
Maximum value = 4 at location 3
./maxloc_implicit
Run on Intel(R) UHD Graphics P630 [0x3e96]
Maximum value is at element 3
./maxloc_buffered
Run on Intel(R) UHD Graphics P630 [0x3e96]
Maximum value is at element 3
./maxloc_usm
Run on Intel(R) UHD Graphics P630 [0x3e96]
Maximum value is at element 3
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
