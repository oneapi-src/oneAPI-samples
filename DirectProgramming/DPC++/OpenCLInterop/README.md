# DPC++ OpenCL&trade; Interoperability Example

The examples here demonstrate how DPC++ can interact with OpenCL&trade;. This enables programmers to incrementally migrate from
OpenCL to DPC++. Two usage scenarios are shown. First is a DPC++ program that compiles and runs an OpenCL kernel. The second program converts OpenCL objects to DPC++.

For more information on migrating from OpenCL to DPC++, see [Migrating OpenCL Designs to DPC++](https://software.intel.com/content/www/us/en/develop/articles/migrating-opencl-designs-to-dpcpp.html).

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, 20; Windows 10
| Hardware                          | Skylake or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler, Intel Devcloud
| What you will learn               | How OpenCL code can interact with DPC++ with the Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 10 minutes

## Purpose
For users who are migrating from OpenCL to DPC++, interoperability allows the migration to take place piecemeal so that the migration of all kernels does not have to take place at the same time.
 
## Key Implementation Details
The common OpenCL to DPC++ conversion scenarios are covered.
1. In dpcpp_with_opencl_kernel.dp.cpp, the DPC++ program compiles and runs an OpenCL kernel. (For this, OpenCL must be set as the backend and not Level 0, the environment variable SYCL_DEVICE_FILTER=OPENCL is used)
2. In dpcpp_with_opencl_objects.dp.cpp, the program converts OpenCL objects (Memory Objects, Platform, Context, Program, Kernel) to DPC++ and execute the program. 

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the Program

> Note: if you have not already done so, set up your CLI 
> environment by sourcing  the setvars script located in 
> the root of your oneAPI installation. 
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh  
> Linux User: . ~/intel/oneapi/setvars.sh  
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat


### On a Linux* System
Perform the following steps:
1. Build the program
	```
    $ mkdir build
    $ cd build
    $ cmake ..
	$ make
	```

2. Run the program (default uses buffers):
    ```
    make run_prog1
    make run_prog2
    ```

3. Clean the program using:
    ```
    make clean
    ```