# `Intrinsics` Sample

The intrinsic samples are designed to show how to utilize the intrinsics supported by the Intel&reg; C++ Compiler in various applications. The src folder contains three .cpp source files, each demonstrating different intrinsics' functionality, including vector operations, complex numbers computations, and FTZ/DAZ flags.

| Optimized for                     | Description
|:---                               |:---
| OS                                | MacOS* Catalina* or newer
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI C++ Compiler Classic
| What you will learn               | How to utilize intrinsics supported by the Intel&reg; oneAPI C++ Compiler Classic
| Time to complete                  | 15 minutes


## Purpose

Intrinsics are assembly-coded functions that allow you to use C++ function calls and variables in place of assembly instructions. Intrinsics are expanded inline, eliminating function call overhead. While providing the same benefits as using inline assembly, intrinsics improve code readability, assist instruction scheduling, and help when debugging. They provide access to instructions that cannot be generated using the C and C++ languages' standard constructs and allow code to leverage performance-enhancing features unique to specific processors.

Further information on intrinsics can be found [here](https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics.html#intrinsics_GUID-D70F9A9A-BAE1-4242-963E-C3A12DE296A1):

## Key Implementation Details

This sample makes use of intrinsic functions to perform common mathematical operations, including:
- Computing a dot product of two vectors
- Computing the product of two complex numbers
The implementations include multiple functions to accomplish these tasks, each one leveraging a different set of intrinsics available to Intel&reg; processors.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building the `Intrinsics` Program

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

Perform the following steps:
1. Build the program using the following `make` commands.
```
$ make (or "make debug" to compile with the -g flag)
```

2. Run the program:
    ```
    make run (or "make debug_run" to run the debug version)
    ```

3. Clean the program using:
    ```
    make clean
    ```

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)

### Application Parameters

These intrinsics samples have relatively few modifiable parameters. However, specific options are available to the user:

1. intrin_dot_sample: Line 35 defines the size of the vectors used in the dot product computation.

2. intrin_double_sample: Lines 244-247 define the values of the two complex numbers used in the computation.

3. intrin_ftz_sample: This sample has no modifiable parameters.

### Example of Output
```
Dot Product computed by C:  4324.000000
Dot Product computed by C + SIMD:  4324.000000
Dot Product computed by Intel(R) SSE3 intrinsics:  4324.000000
Dot Product computed by Intel(R) AVX2 intrinsics:  4324.000000
Dot Product computed by Intel(R) AVX intrinsics:  4324.000000
Dot Product computed by Intel(R) MMX(TM) intrinsics:  4324
Complex Product(C):             23.00+ -2.00i
Complex Product(Intel(R) AVX2): 23.00+ -2.00i
Complex Product(Intel(R) AVX):  23.00+ -2.00i
Complex Product(Intel(R) SSE3): 23.00+ -2.00i
Complex Product(Intel(R) SSE2): 23.00+ -2.00i
FTZ is set.
DAZ is set.
```
