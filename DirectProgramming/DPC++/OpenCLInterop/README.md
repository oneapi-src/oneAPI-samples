# DPC++ OpenCL Interoperability Example

The examples here demonstrate how DPC++ can interact with OpenCL. This enables programmers to incrementally migrate from
OpenCL to DPC++. Two usage scenarios are shown.
1. Shows a DPC++ program that compiles and runs an OpenCL kernel. (For this, OpenCL must be set as the backend and not Level 0)
	Use the environment variable SYCL_DEVICE_FILTER=OPENCL
2. Show how to convert OpenCL objects (Memory Objects, Platform, Context, Program, Kernel) to DPC++ and execute the program. 


| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, 20; Windows 10
| Hardware                          | Skylake or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler, Intel Devcloud
| What you will learn               | How OpenCL code can interact with DPC++ with the Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 10 minutes

## Purpose
For users who are migrating from OpenCL to DPC++, interoperability allows the migration to take place piecemeal.
 
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
	make
	```

2. Run the program (default uses buffers):
    ```
    make Run
    ```

3. Clean the program using:
    ```
    make clean
    ```
