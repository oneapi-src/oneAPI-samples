# sepia-filter sample
Sepia filter is a program that converts a color image to a Sepia tone image which is a monochromatic image with a distinctive Brown Gray color. The program works by offloading the compute intensive conversion of each pixel to Sepia tone and is implemented using DPC++ for CPU and GPU.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | The Sepia Filter sample demonstrates the following using the Intel&reg; oneAPI DPC++/C++ Compiler <ul><li>Writing a custom device selector class</li><li>Offloading compute intensive parts of the application using both lamba and functor kernels</li><li>Measuring kernel execution time by enabling profiling</li></ul>
| Time to complete                  | 20 minutes

## Purpose
Sepia filter is a DPC++ application that accepts a color image as an input and converts it to a sepia tone image by applying the sepia filter coefficients to every pixel of the image. The sample demonstrates offloading the compute intensive part of the application which is the processing of individual pixels to an accelerator with the help of lambda and functor kernels.  The sample also demonstrates the usage of a custom device selector which sets precedence for a GPU device over other available devices on the system.
The device selected for offloading the kernel is displayed in the output along with the time taken to execute each of the kernels. The application also outputs a sepia tone image of the input image.

## Key implementation details
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.  In addition, this sample demonstrates implementation of a custom device selector by over writing the SYCL device selector class, offloading computation using both lambda and functor kernels and using event objects to time command group execution by enabling profiling.
 
## License  
This code sample is licensed under MIT license 

## Building the Program for CPU and GPU

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel&reg; oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System

Perform the following steps:


1.  Build the program using the following <code> cmake </code> commands.
```
    $ cd sepia-filter  
    $ mkdir build  
    $ cd build  
    $ cmake ../.  
    $ make
```

2.  Run the program <br>
```
    $ make run

```

### On a Windows* System Using Visual Studio* version 2017 or Newer

* Build the program using VS2017 or VS2019: Right click on the solution file and open using either VS2017 or VS2019 IDE. Right click on the project in Solution explorer and select Rebuild. From top menu select Debug -> Start without Debugging.
* Build the program using MSBuild: Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019". Run - MSBuild sepia-filter.sln /t:Rebuild /p:Configuration="Release"


## Running the sample

### Application Parameters
The Sepia-filter application expects a png image as an input parameter. The application comes with some sample images in the input folder. One of these is specified as the default input image to be converted in the cmake file and the default output image is generated in the same folder as the application.

### Example Output
```
Loaded image with a width of 3264, a height of 2448 and 3 channels
Running on Intel(R) Gen9
Submitting lambda kernel...
Submitting functor kernel...
Waiting for execution to complete...
Execution completed
Lambda kernel time: 10.5153 milliseconds
Functor kernel time: 9.99602 milliseconds
Sepia tone successfully applied to image:[input/silverfalls1.png]
```
## Known Limitations

Due to a known issue in the Level0 driver sepia-filter fails with default Level0 backend. A workaround is in place to enable OpenCL backend.