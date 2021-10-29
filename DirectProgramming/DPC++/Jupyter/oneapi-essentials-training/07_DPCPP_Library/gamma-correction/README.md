# Parallel STL 'Gamma Correction' Sample
Gamma correction is a nonlinear operation used to encode and decode the luminance of each pixel of an image. This sample demonstrates use of Parallel STL algorithms from Intel&reg; oneAPI DPC++ Library (oneDPL) to facilitate offload to devices.

| Optimized for                   | Description                                                                      |
|---------------------------------|----------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04, Windows 10                                                 |
| Hardware                        | Skylake with GEN9 or newer                                                       |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL)   |
| What you will learn             | How to offload the computation to GPU using Intel&reg; oneAPI DPC++ Library      |
| Time to complete                | At most 5 minutes                                                                |

## Purpose

Gamma correction uses nonlinear operations to encode and decode the luminance of each pixel of an image. See https://en.wikipedia.org/wiki/Gamma_correction for more information.
It does so by creating a fractal image in memory and performs gamma correction on it with `gamma=2`.
A device policy is created and passed to the `std::for_each` Parallel STL algorithm.
This example demonstrates how to use Parallel STL algorithms, Parallel STL is a component of Intel&reg; oneAPI DPC++ Library (oneDPL).

Parallel STL is an implementation of the C++ standard library algorithms with support for execution policies, as specified in ISO/IEC 14882:2017 standard, commonly called C++17. The implementation also supports the unsequenced execution policy specified in the final draft for the C++ 20 standard (N4860).

Parallel STL offers efficient support for both parallel and vectorized execution of algorithms for Intel&reg; processors. For sequential execution, it relies on an available implementation of the C++ standard library. The implementation also supports the unsequenced execution policy specified in the final draft for the next version of the C++ standard and DPC++ execution policy specified in the oneDPL Spec (https://spec.oneapi.com/versions/latest/elements/oneDPL/source/pstl.html).

## Key Implementation Details

`std::for_each` Parallel STL algorithms are used in the code.

## License

This code sample is licensed under MIT license.

## Building the 'Gamma Correction' Program for CPU and GPU

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel&reg; oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:

1. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

2. Run the program:
```
    $ make run
```

3. Clean the program using:
```
    $ make clean
```

### On a Windows* System Using Visual Studio* Version 2017 or Newer
* Build the program using VS2017 or VS2019. Right click on the solution file and open using either VS2017 or VS2019 IDE. Right click on the project in Solution explorer and select Rebuild. From top menu select Debug -> Start without Debugging.
* Build the program using MSBuild. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019". Run - MSBuild gamma-correction.sln /t:Rebuild /p:Configuration="Release"

## Running the Sample
### Example of Output

The output of the example application is a BMP image with corrected luminance. Original image is created by the program.
```
success
Run on Intel(R) Gen9
Original image is in the fractal_original.bmp file
Image after applying gamma correction on the device is in the fractal_gamma.bmp file
```
