# `vector-add` Sample

Vector Add is the equivalent of a Hello, World! sample for data parallel programs. Building and running the sample verifies that your development environment is setup correctly and demonstrates the use of the core features of DPC++.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler  
  
## Purpose
The `vector-add` is a simple program that adds two large vectors of integers and verifies the results. This program is implemented using C++ and Data Parallel C++ (DPC++) for Intel(R) CPU and accelerators.

In this sample, you can learn how to use the most basic code in C++ language that offloads computations to a GPU using the DPC++ language. This includes using Unified Shared Memory (USM) and buffers. USM requires explicit wait for the asynchronous kernel's computation to complete.  Buffers, at the time they go out of scope, bring main memory in sync with device memory implicitly; the explicit wait on the event is not required as a result. This sample provides examples of both implementations for simple side by side review (Windows sample only supports USM).

The code will attempt first to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected. If successful, the name of the offload device and a success message are displayed. And, your development environment is setup correctly!

In addition, you can target an FPGA device using build scripts described below.  If you do not have FPGA hardware, the sample will run in emulation mode, which includes static optimization reports for design analysis.

## Key Implementation Details 
The basic DPC++ implementation explained in the code includes device selector, USM, buffer, accessor, kernel, and command groups.

## License  
This code sample is licensed under MIT license. 

## Building the `vector-add` Program for CPU and GPU 

### On a Linux* System
Perform the following steps:
1. Build the program using the following `make` commands (default uses buffers):
    ```
    make all
    ```
> Note: for USM use `make build_usm`

2. Run the program using:
    ```
    make run
    ```
> Note: for USM use `make run_usm`

3. Clean the program using:  
    ```
    make clean 
    ```

### On a Windows* System Using a Command Line Interface
1. Select **Programs** > **Intel oneAPI 2021** > **Intel oneAPI Command Prompt** to launch a command window.
2. Build the program using the following `nmake` commands:
    ```
    nmake -f Makefile.win
    ```
> Note: for USM use `nmake -f Makefile.win build_usm`

3. Run the program using:
    ```
    nmake -f Makefile.win run
    ```
> Note: for USM use `nmake -f Makefile.win run_usm`

4. Clean the program using:
    ```
    nmake -f Makefile.win clean
    ```


### On a Windows* System Using Visual Studio* Version 2017 or Newer
Perform the following steps:
1. Launch the Visual Studio* 2017.
2. Select the menu sequence **File** > **Open** > **Project/Solution**. 
3. Locate the `vector-add` folder.
4. Select the `vector-add.sln` file.
5. Select the configuration 'Debug' or 'Release'  
6. Select **Project** > **Build** menu option to build the selected configuration.
7. Select **Debug** > **Start Without Debugging** menu option to run the program.

## Building the `vector-add` Program for Intel(R) FPGA

### On a Linux* System

Perform the following steps:

1. Clean the `vector-add` program using:
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
    * Generate static optimization reports for design analysis. Path to the reports is `vector-add_report.prj/reports/report.html`
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
3. Locate the `vector-add` folder.
4. Select the `vector-add.sln` file.
5. Select the configuration 'Debug-fpga' that have the necessary project settings already below:
 
            Under the 'Project Property' dialog:
 
     a. Select the **DPC++** tab.
     b. In the **General** subtab, the **Perform ahead of time compilation for the FPGA** setting is set to **Yes**.
     c. In the **Preprocessor** subtab, the **Preprocessor Definitions" setting has **FPGA_EMULATOR=1** added.
     d. Close the dialog.
 
6. Select **Project** > **Build** menu option to build the selected configuration.
7. Select **Debug** > **Start Without Debugging** menu option to run the program.

## Running the Sample
### Application Parameters
There are no editable parameters for this sample.

### Example of Output
<pre>
Running on device:        Intel(R) Gen(R) HD Graphics NEO
Vector size: 10000
[0]: 0 + 0 = 0
[1]: 1 + 1 = 2
[2]: 2 + 2 = 4
...
[9999]: 9999 + 9999 = 19998
Vector add successfully completed on device.
</pre>
