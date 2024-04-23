# Parallel STL 'Gamma Correction' Sample
Gamma correction is a nonlinear operation used to encode and decode the luminance of each pixel of an image. This sample demonstrates use of Parallel STL algorithms from Intel&reg; oneAPI DPC++ Library (oneDPL) to facilitate offload to devices.

| Optimized for                   | Description                                                                      |
|:--- |:---|
| OS                              | Linux* Ubuntu* 18.04 <br> Windows*10                                                 |
| Hardware                        | Skylake with GEN9 or newer                                                       |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler <br> Intel&reg; oneAPI DPC++ Library (oneDPL)   |
| What you will learn             | How to offload the computation to GPU using OneDPL      |
| Time to complete                | At most 5 minutes                                                                |

## Purpose

Gamma correction uses nonlinear operations to encode and decode the luminance of each pixel of an image. (See https://en.wikipedia.org/wiki/Gamma_correction for more information.)

It does so by creating a fractal image in memory and performs gamma correction on it with `gamma=2`.

A device policy is created and passed to the `std::for_each` Parallel STL algorithm.
This example demonstrates how to use Parallel STL algorithms, Parallel STL is a component of Intel&reg; oneAPI DPC++ Library (oneDPL).

Parallel STL is an implementation of the C++ standard library algorithms with support for execution policies, as specified in ISO/IEC 14882:2017 standard, commonly called C++17. The implementation also supports the unsequenced execution policy specified in the final draft for the C++ 20 standard (N4860).

Parallel STL offers efficient support for both parallel and vectorized execution of algorithms for Intel&reg; processors. For sequential execution, it relies on an available implementation of the C++ standard library. The implementation also supports the unsequenced execution policy specified in the final draft for the next version of the C++ standard and execution policy specified in the oneDPL Spec (https://spec.oneapi.com/versions/latest/elements/oneDPL/source/pstl.html).

## Key Implementation Details

`std::for_each` Parallel STL algorithms are used in the code.

## Building the 'Gamma Correction' Program for CPU and GPU

### Setting Environment Variables
When working with the Command Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation.
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
> Microsoft Visual Studio:
> - Open a command prompt window and execute `setx SETVARS_CONFIG " "`. This only needs to be set once and will automatically execute the `setvars` script every time Visual Studio is launched.
>
>For more information on environment variables, see "Use the setvars Script" for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

If you wish to fine tune the list of components and the version of those components, use
a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI Toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### Running Samples In Intel® DevCloud
If running a sample in the Intel® DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel&reg; oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### On a Linux* System
Perform the following steps:

1. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

2. Run the program:
```
    $ make run
```

3. Clean the program using:
```
    $ make clean
```

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### On a Windows* System Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild gamma-correction.sln /t:Rebuild /p:Configuration="Release"`

## Running the Sample
### Example of Output

The output of the example application is a BMP image with corrected luminance. Original image is created by the program.
```
success
Run on Intel(R) Gen9
Original image is in the fractal_original.bmp file
Image after applying gamma correction on the device is in the fractal_gamma.bmp file
```
## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
