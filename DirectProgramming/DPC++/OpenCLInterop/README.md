# DPC++ OpenCL&trade; Interoperability Example

This examples demonstrate how DPC++ can interact with OpenCL&trade;. This code
sample illustrates how to incrementally migrate from OpenCL to DPC++. Two
usage scenarios are shown: the first is a DPC++ program that compiles and runs
an OpenCL kernel; the second program converts OpenCL objects to DPC++.

For more information on migrating from OpenCL to DPC++, see
[Migrating OpenCL Designs to DPC++](https://software.intel.com/content/www/us/en/develop/articles/migrating-opencl-designs-to-dpcpp.html).

| Optimized for        | Description
|:---                  |:---
| OS                   | Linux* Ubuntu* 18.04, 20
| Hardware             | Skylake or newer
| Software             | Intel&reg; oneAPI DPC++/C++ Compiler, Intel Devcloud
| What you will learn  | How OpenCL code can interact with DPC++ with the Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete     | 10 minutes

## Purpose

For users migrating from OpenCL to DPC++, interoperability allows the
migration to take place piecemeal. It is not necessary for migration of all
existing OpenCL kernels occur simultaneously.

## Key Implementation Details

Common OpenCL to DPC++ conversion scenario is demonstrated.

1. In `dpcpp_with_opencl_objects.dp.cpp`, the program converts OpenCL objects
   (Memory Objects, Platform, Context, Program, Kernel) to DPC++ and execute the
   program.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the Program

> Note: if you have not already done so, set up your CLI
> environment by sourcing  the setvars script located in
> the root of your oneAPI installation.
>
> Linux sudo: . /opt/intel/oneapi/setvars.sh
> Linux user: . ~/intel/oneapi/setvars.sh
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat

### Running Samples In DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the
compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.
For more information, see the Intel® oneAPI Base Toolkit Get Started Guide
(https://devcloud.intel.com/oneapi/get_started/baseToolkitSamples/)


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System

Perform the following steps:

1. Build the program
	```
    $ mkdir build
    $ cd build
    $ cmake ..
	$ make
	```

2. Run the program:
    ```
    make run_prog1
    make run_prog2
    ```

3. Clean the program using:
    ```
    make clean
    ```

### Output Example
```
Device: Intel(R) HD Graphics 630 [0x5912]
PASSED!
Built target run_prog1

Kernel Loading Done
Platforms Found: 3
Using Platform: Intel(R) FPGA Emulation Platform for OpenCL(TM)
Devices Found: 1
Device: Intel(R) FPGA Emulation Device
Passed!
Built target run_prog2
```
