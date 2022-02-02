# OpenMP Offload Features

These examples demonstrate some of the new OpenMP Offload features supported
by the Intel&reg; oneAPI DPC++/C++ Compiler.

For more information on the compiler see the
[Intel&reg oneAPI DPC++/C++ Compiler Landing Page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html).

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04, 20
| Hardware             | Skylake with GEN9 or newer
| Software             | Intel&reg; oneAPI DPC++/C++ Compiler, Intel Devcloud
| What you will learn  | Understand some of the new OpenMP Offload features supported by the Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete     | 15 minutes


## Purpose

For developers to understand some of the new OpenMP Offload features supported
by the Intel oneAPI DPC++/C++ Compiler.


## Key Implementation Details

The table below shows the designs and the demonstrated feature(s).

| Design                           | Feature(s) Utilized
| :---                             |:---
| class_member_functor             | Usage of functor in an OpenMP offload region
| function_pointer                 | Function called through a function pointer in an offload region
| user_defined_mapper              | Usage of the user defined mapper feature in target region map clauses
| usm_and_composabilty_with_dpcpp  | Unified shared memory and composability with DPC++


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)



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


## Building the Program

> Note: if you have not already done so, set up your CLI
> environment by sourcing  the setvars script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
> Linux User: . ~/intel/oneapi/setvars.sh
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat


### Running Samples In DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the
compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.
For more information, see the Intel® oneAPI Base Toolkit Get Started Guide
(https://devcloud.intel.com/oneapi/get_started/baseToolkitSamples/)


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
    make run_prog3
    make run_prog4
    ```

3. Clean the program using:
    ```
    make clean
    ```

### Example of Output

```
6 8 10 12 14
Done ......
Built target run_prog1

Scanning dependencies of target run_prog2
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

Scanning dependencies of target run_prog3
In :   1   2   4   8
Out:   2   4   8  16
In :   1   2   4   8
Out:   2   4   8  16
In :   1   2   4   8  16  32  64 128
Out:   2   4   8  16  32  64 128 256
In :   1   2   4   8  16  32  64 128
Out:   2   4   8  16  32  64 128 256
Built target run_prog3

Scanning dependencies of target run_prog4
SYCL: Running on Intel(R) HD Graphics 630 [0x5912]
SYCL and OMP memory: passed
OMP and OMP memory:  passed
OMP and SYCL memory: passed
SYCL and SYCL memory: passed
Built target run_prog4
```
