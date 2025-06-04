# `OpenMP Offload` Sample

The `OpenMP Offload` sample demonstrates some of the new OpenMP offload features supported
by the Intel® oneAPI DPC++/C++ Compiler.

For more information, see [Intel® oneAPI DPC++/C++ Compiler](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html).

| Area                 | Description
|:---                  |:---
| What you will learn  | Understand some of the new OpenMP Offload features
| Time to complete     | 15 minutes

## Purpose

For developers to understand some of the new OpenMP Offload features supported
by the Intel® compiler.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04, 20.04
| Hardware             | Skylake with GEN9 or newer
| Software             | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

The table below shows the designs and the demonstrated features.

| Source                                 | Demonstrates
| :---                                   |:---
| `class_member_functor.cpp`             | Using a functor in an OpenMP offload region
| `function_pointer.cpp`                 | Function called through a function pointer in an offload region (currently only for CPU target)
| `user_defined_mapper.cpp`              | User defined mapper feature in target region map clauses
| `usm_and_composabilty_with_dpcpp.cpp`  | Unified shared memory and composability with SYCL*

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.


## Build the `OpenMP Offload` Programs

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).


### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `OpenMP Offload` Program

### On Linux*

1. Run the programs.
   ```
   make run_prog1
   make run_prog2
   make run_prog3
   make run_prog4
   ```

2. Clean the program. (Optional)
   ```
   make clean
   ```

## Example Output

### Prog1 Example Output
```
6 8 10 12 14
Done ......
Built target run_prog1
```

### Prog2 Example Output

```
called from device, y = 100
called from device, y = 103
called from device, y = 114
called from device, y = 106
called from device, y = 109
called from device, y = 112
called from device, y = 115
called from device, y = 107
called from device, y = 104
called from device, y = 101
called from device, y = 102
called from device, y = 110
called from device, y = 113
called from device, y = 105
called from device, y = 111
called from device, y = 108
Output x = 1720
Built target run_prog2
```

### Prog3 Example Output

```
In :   1   2   4   8
Out:   2   4   8  16
In :   1   2   4   8
Out:   2   4   8  16
In :   1   2   4   8  16  32  64 128
Out:   2   4   8  16  32  64 128 256
In :   1   2   4   8  16  32  64 128
Out:   2   4   8  16  32  64 128 256
Built target run_prog3
```

### Prog4 Example Output

```
SYCL: Running on Intel(R) HD Graphics 630 [0x5912]
SYCL and OMP memory: passed
OMP and OMP memory:  passed
OMP and SYCL memory: passed
SYCL and SYCL memory: passed
Built target run_prog4
```

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt)
for details.

Third party program licenses are at
[third-party-programs.txt](third-party-programs.txt).
