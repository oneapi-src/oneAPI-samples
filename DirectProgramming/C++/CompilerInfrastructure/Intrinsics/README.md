# `Intrinsics` Sample

This `Intrinsics` sample demonstrates how to use intrinsics supported by the Intel® C++ Compiler.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to use intrinsics supported by the  Intel® C++ Compiler
| Time to complete         | 15 minutes


## Purpose

Intrinsics are assembly-coded functions that allow you to use C++ function calls and variables in place of assembly instructions. Intrinsics expand inline, eliminating function call overhead. While providing the same benefits as using inline assembly, intrinsics improve code readability, assist instruction scheduling, and help when debugging. They provide access to instructions that cannot be generated using C and C++ language standard constructs and allow code to use performance-enhancing features unique to specific processors.

The `src` folder contains three .cpp source files, each demonstrating different functionality, including vector operations, complex numbers computations, and FTZ/DAZ flags.

You can find detailed information in the *Intrinsics* section of the [Intel® C++ Compiler Classic Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/).

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                  | Ubuntu* 18.04
| Hardware            | Skylake with GEN9 or newer
| Software            | Intel® C++ Compiler

## Key Implementation Details

This sample uses intrinsic functions to perform common mathematical operations, including:
- Computing a dot product of two vectors
- Computing the product of two complex numbers

The implementations include multiple functions to accomplish these tasks, each one using a different set of intrinsics available to the Intel® processor family.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Intrinsics` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Build the program.
   ```
   make
   ```
   Alternatively, build the debug version, which compiles with the `-g` option.
   ```
   make debug
   ```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Intrinsics` Program

### Configurable Parameters

The sample source files contain some configurable parameters.

- `intrin_dot_sample`: Modify line 35 to change the size of the vectors used in the dot product computation.
- `intrin_double_sample`: Modify lines 244-247 to define the values of the two complex numbers used in the computation.
- `intrin_ftz_sample`: This sample has no configurable parameters.

1. Run the program:
    ```
    make run
    ```
    Alternatively, run the debug version.
    ```
    make debug_run
    ```
2. Clean the program. (Optional)
    ```
    make clean
    ```

### Example Output
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

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
