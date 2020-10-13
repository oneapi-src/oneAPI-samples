# Complex Multiplication sample

complex multiplication is a program that multiplies two large vectors of 
Complex numbers in parallel and verifies the results. It also implements 
custom device selector to target a specific vendor device. This program is 
implemented using C++ and DPC++ language for Intel CPU and accelerators. 
The Complex class is a custom class and this program shows how we can use 
custom types of classes in a DPC++ program
  
| Optimized for                       | Description
ii|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | Using custom type classes and offloads complex number computations to GPU using Intel DPC++
| Time to complete                  | 15 minutes  
  

## Purpose	

Complex multiplication multiplies two vectors with complex numbers. The 
code will attempt to run the calculation on both the GPU and CPU, and then 
verifies the results. The size of the computation can be adjusted for 
heavier workloads. If successful, the name of the offload device and a 
success message are displayed.

This sample uses buffers to manage memory. For more information regarding
different memory management options, refer to the vector_add sample.

complex multiplication includes C++ implementations of both Data Parallel 
(DPC++). 

This program shows how to create a custom device selector and to target 
GPU or CPU of a specific vendor. The program also shows how to pass in a 
vector of custom Complex class objects that you can do the parallel 
executions on the device. The device used for compilation is displayed in 
the output.


## Key implementation details 

This program shows how we can use custom types of classes in a DPC++ 
program and explains the basic DPC++ implementation including device 
selector, buffer, accessor, kernel and command group.  


## License  

This code sample is licensed under MIT license. 

## Building the complex_mult Program for CPU and GPU 

Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify 
the compute node (CPU, GPU, FPGA) as well whether to run in batch or 
interactive mode. For more information see the Intel® oneAPI Base Toolkit 
Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System 
   * Build the program using Make  
    make all  

   * Run the program  
    make run  

   * Clean the program  
    make clean 

### On a Windows* System Using Visual Studio* Version 2017 or Newer
* Build the program using VS2017 or VS2019
      Right click on the solution file and open using either VS2017 or 
      VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.

* Build the program using MSBuild
      Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native 
      Tools Command Prompt for VS2019"
      Run - MSBuild complex_mult.sln /t:Rebuild /p:Configuration="debug"


## Running the Sample

### Application Parameters
There are no editable parameters for this sample.

### Example of Output

```
	Target Device: Intel(R) Gen9

	****************************************Multiplying Complex numbers in Parallel********************************************************
	[0] (2 : 4i) * (4 : 6i) = (-16 : 28i)
	[1] (3 : 5i) * (5 : 7i) = (-20 : 46i)
	[2] (4 : 6i) * (6 : 8i) = (-24 : 68i)
	[3] (5 : 7i) * (7 : 9i) = (-28 : 94i)
	[4] (6 : 8i) * (8 : 10i) = (-32 : 124i)
	...
	[9999] (10001 : 10003i) * (10003 : 10005i) = (-40012 : 200120014i)
	Complex multiplication successfully run on the device
   
    ```
