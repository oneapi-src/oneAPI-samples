# `OpenCL™ Interoperability Sample`
This sample illustrates incremental migration from OpenCL™ to SYCL* compliant code in two usage scenarios:

1. A program that compiles and runs an OpenCL kernel using SYCL.
2. A program that converts OpenCL objects using SYCL standards. (For more information, read [Migrating OpenCL™ Designs to SYCL](https://software.intel.com/content/www/us/en/develop/articles/migrating-opencl-designs-to-dpcpp.html).)

| Property               | Description
|:---                    |:---
| What you will learn    | How OpenCL code can interact with SYCL using the Intel® oneAPI DPC++/C++ Compiler
| Time to complete       | 10 minutes

## Purpose
For users migrating from OpenCL to SYCL, the sample illustrates interoperability that enables incremental migration. Simultaneous migration of existing OpenCL kernels is unnecessary.

## Prerequisites
| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04
| Hardware             | Skylake or newer
| Software             | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
Common OpenCL to SYCL conversion scenarios are illustrated in the code; in `sycl_with_opencl_objects.dp.cpp`, the program converts OpenCL objects (Memory Objects, Platform, Context, Program, Kernel) to conform with SYCL standards and executes the program.

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `OpenCL Interoperability Sample` Program
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
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
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

> **Note**: You might see deprecation warnings.

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Run the `OpenCL Interoperability Sample` Program
### On Linux
1. Run the program.
   ```
   make run
   ```
2. Clean the program. (Optional)
   ```
   make clean
   ```

### Run `OpenCL Interoperability Sample` in Intel® DevCloud
When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

## Output Example
```
Kernel Loading Done
Platforms Found: 3
Using Platform: Intel(R) FPGA Emulation Platform for OpenCL(TM)
Devices Found: 1
Device: Intel(R) FPGA Emulation Device
Passed!
Built target run
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).