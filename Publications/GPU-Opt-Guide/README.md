# oneAPI GPU Optimization Guide Examples
This repository accompanies the
[*oneAPI GPU Optimization Guide* Developer Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top.html).

### Get the Examples
You can get the examples in either of two methods:
- Use Git to clone the [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples) GitHub repository on your system.
- Download the files by selecting **Code** > **Download ZIP** from [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples) GitHub repository.

## Purpose
This repository contains the example sources mentioned in the *oneAPI GPU Optimization Guide* Developer Guide along with the build scripts for building and running the examples.

In tandem with the information found in the guide, the examples in the repository allow programmers to experiment with the examples and understand the techniques presented in the *oneAPI GPU Optimization Guide*.

## Prerequisites
To build and use these examples, you will need to one or more of the following toolkits:

- [Intel® oneAPI Base Toolkit (Base Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html)

- [Intel® oneAPI HPC Toolkit (HPC Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit.html)

- [Intel® oneAPI IoT Toolkit (IoT Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/iot-toolkit.html)

> **Note**: You will need the HPC Kit to build the Fortran examples in this repository. The Intel® Fortran Compiler is included in the HPC Kit.

Alternatively, you can build most of the toolchain from the sources in the [Intel Project for LLVM* technology](https://github.com/intel/llvm) GitHub repository.

| Minimum Requirement   | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04
| Hardware              | Skylake with GEN9 or newer
| Software              | Intel® oneAPI DPC++/C++ Compiler <br> Intel® Fortran Compiler

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

> **Note**: You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

> **Note**: If you want to select specific components or the version of those components, use a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

## Build the Examples
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Include Files
After installing the toolkits, the include folder is at `%ONEAPI_ROOT%/dev-utilities/latest/include` on your development system. You might need to use some of the resources from this location to build the examples.

Alternatively, you can get the common resources from the [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples/tree/master/common) GitHub repository.

### On Linux*
1. Change to the `oneAPI-samples/Publications/GPU-Opt-Guide` directory on your system.
2. Build the example program.
   ```sh
   mkdir -p build
   cd build
   cmake ..
   make
   ```
## Run the Examples
### On Linux
1. Ensure you are in the correct example build directory.
2. Run the example.
    ```sh
    make test
    ```
3. Clean the project files. (Optional)
   ```
   make clean
   ```

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
