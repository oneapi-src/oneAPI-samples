# `simple-add-dpc++` Sample

`simple-add-dpc++` provides the simplest example of DPC++ while providing an example of using both buffers and Unified Shared Memory.   

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler



## Purpose
The `simple-add-dpc++` is a simple program that adds two large vectors of integers and verifies the results. This program is implemented using C++ and Data Parallel C++ (DPC++) for Intel(R) CPU and accelerators.

In this sample, you can learn how to use the most basic code in C++ language that offloads computations to a GPU using the DPC++ language. This includes using Unified Shared Memory (USM) and buffers. USM requires explicit wait for the asynchronous kernel's computation to complete.  Buffers, at the time they go out of scope, bring main memory in sync with device memory implicitly; the explicit wait on the event is not required as a result. This sample provides examples of both implementations for simple side by side review.

The code will attempt first to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected. If successful, the name of the offload device and a success message are displayed. And, your development environment is setup correctly!

## Key Implementation Details 
The basic DPC++ implementation explained in the code includes device selector, USM, buffer, accessor, kernel, and command groups.

## License  
This code sample is licensed under MIT license. 

## Building the `simple add DPC++` Program for CPU and GPU 

## Include Files
The include folder is located at "%ONEAPI_ROOT%\dev-utilities\latest\include" on your development system.

### On a Linux* System
Perform the following steps:
1. Build the `simple-add-dpc++` program using the following make commands (default uses USM): 
    ```
    make all
    ```
> Note! To build with buffers use: `make build_buffers`

2. Run the program using:  
    ```
    make run
    ```
> Note! To run with buffers use: `make run_buffers`

3. Clean the program using:  
    ```
    make clean 
    ```

### On a Windows* System Using a Command Line Interface
1. Select **Programs** > **Intel oneAPI 2021** > **Intel oneAPI Command Prompt** to launch a command window.
2. Build the program using the following `nmake` commands (Windows supports USM only):

    ```
    nmake -f Makefile.win
    ```

3. Run the program using:  
    ```
    nmake -f Makefile.win run
    ```

4. Clean the program using:  
    ```
    nmake -f Makefile.win clean 
    ```
	

### On a Windows* System Using Visual Studio* Version 2017 or Newer
Perform the following steps:
1. Launch the Visual Studio* 2017.
2. Select the menu sequence **File** > **Open** > **Project/Solution**. 
3. Locate the `simple-add` folder.
4. Select the `simple-add.sln` file.
5. Select the configuration 'Debug' or 'Release'  
6. Select **Project** > **Build** menu option to build the selected configuration.
7. Select **Debug** > **Start Without Debugging** menu option to run the program.


## Building the `simple-add` Program for Intel(R) FPGA

### On a Linux* System

Perform the following steps:

1. Clean the `simple-add` program using:
    ```
    make clean -f Makefile.fpga
    ```

2. Based on your requirements, you can perform the following:
   * Build and run for FPGA emulation using the following commands:
    ```
    make fpga_emu -f Makefile.fpga
    make run_emu -f Makefile.fpga
    ```
    * Build and run for FPGA hardware. 
      **NOTE:** The hardware compilation takes a long time to complete.
    ```
    make hw -f Makefile.fpga
    make run_hw -f Makefile.fpga
    ```
    * Generate static optimization reports for design analysis. Path to the reports is `simple-add_report.prj/reports/report.html`
    ```
    make report -f Makefile.fpga
    ```

### On a Windows* System Using a Command Line Interface
Perform the following steps:

**NOTE:** On a Windows* system, you can only compile and run on the FPGA emulator. Generating an HTML optimization report and compiling and running on the FPGA hardware are not currently supported.

1. Select **Programs** > **Intel oneAPI 2021** > **Intel oneAPI Command Prompt** to launch a command window.
2. Build the program using the following `nmake` commands:
   ```
   nmake -f Makefile.win.fpga clean
   nmake -f Makefile.win.fpga
   nmake -f Makefile.win.fpga run
   ```
               
### On a Windows* System Using Visual Studio* Version 2017 or Newer
Perform the following steps:
1. Launch the Visual Studio* 2017.
2. Select the menu sequence **File** > **Open** > **Project/Solution**.
3. Locate the `simple-add` folder.
4. Select the `simple-add.sln` file.
5. Select the configuration 'Debug-fpga'
6. Select **Project** > **Build** menu option to build the selected configuration.
7. Select **Debug** > **Start Without Debugging** menu option to run the program.

## Running the Sample
### Application Parameters
There are no editable parameters for this sample.

### Example of Output
<pre>simple-add output snippet changed to:
Running on device:        Intel(R) Gen9 HD Graphics NEO
Array size: 10000
[0]: 0 + 100000 = 100000
[1]: 1 + 100000 = 100001
[2]: 2 + 100000 = 100002
...
[9999]: 9999 + 100000 = 109999
Successfully completed on device.</pre>
