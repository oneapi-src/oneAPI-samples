# CMake Template Examples for OneAPI

The CMake projects in this code repository demonstrate use of CMake to build
simple programs with OneAPI compilers for various common scenarios.


| Area                      | Description
|:---                       |:---
| What you will learn       | How to use CMake to build projects with OneAPI compilers.
| Time to complete          | 10 minutes
| Category                  | Tutorial

## Purpose

The CMake projects in this code repository demonstrate use of CMake to build
simple programs with OneAPI Compilers for various common scenarios. Examples are divided
-into directories for C, C++, and Fortran. SYCL examples are included at the
top level. SYCL examples are implemented entirely in C++.

## Prerequisites

| Optimized for             | Description
|:---                       |:---
| OS                        | Windows, Linux
| Hardware                  | N/A
| Software                  | CMake, OneAPI Compilers

## Key Implementation Details


Each language specific directory contains a one or more samples with examples on how to use cmake with OneAPI (with examples using IPO/OpenMP).

The examples in this directory are structured as a collection of independent
projects, rather as a large single project. This way, any example can be copied
into a separate subdirectory, compiled, run, and used as the basis for a new project.

The top level CMakeLists.txt includes projects in all the sub-directories.  To
build all of the examples, create a build directory and generate the project as
usual.

>**Note**: CMake support for GNU style OneAPI driver `icpx` on Windows is not yet available in CMake 3.25.
>**Note**: For comprehensive information about oneAPI programming, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)
>**Note**: For comprehensive information about CMake with OneAPI Compilers, see the *[Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference)*. (Use search or the table of contents to find relevant information quickly. OR, Navigate to Compiler Setup -> Use the Command Line -> Use the Cmake with the compiler)


## Set Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

Ensure the availability of latest CMake binaries.

## Build the Cmake example Sample

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables.
> Set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation every time you open a new terminal window.
> This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).
Use these commands to run the design, depending on your OS.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ cmake --build .
   ```

### On Windows*


1. Change to the sample directory.
2. Build the program.
   ```
    $ mkdir build
    $ cd build
    $ cmake -GNinja ..
    $ cmake --build .
   ```
>**Note**: Currently, only Ninja generators are supported on Windows.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```



Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

© Intel Corporation.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
