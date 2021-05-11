# DPC++ OpenCL Interoperability Examples 
  
## Requirements
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Intel Devcloud
  
## Purpose
The samples here show how to incrementally migrate OpenCL to DPC++.
1. Shows a DPC++ program that compiles and runs an OpenCL kernel. (For this, OpenCL must be set as the backend and not Level 0)
    - Use the environment variable SYCL_DEVICE_FILTER=OPENCL
3. Show how to convert OpenCL objects (Memory Objects, Platform, Context, Program, Kernel) to DPC++ and execute the program. 

## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
