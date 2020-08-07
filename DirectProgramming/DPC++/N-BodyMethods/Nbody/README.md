# nbody sample
An N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity. This nbody sample code is implemented using C++ and DPC++ language for Intel CPU and GPU. 
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (Beta) 

## License  
This code sample is licensed under MIT license. 

### Original source
Source: https://github.com/fbaru-dev/particle-sim 
License: MIT license 

## How to Build  

### on Linux  
Build Nbody program  

    cd Nbody &&  
    mkdir build &&  
    cd build &&  
    cmake ../. &&  
    make  

Run the program  

    make run  

Clean the program 

    make clean  

### on Windows

Build the program using VS2017 or VS2019

      Right click on the solution file and open using either VS2017 or VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.

Build the program using MSBuild

      Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
      Run - MSBuild Nbody.sln /t:Rebuild /p:Configuration="Release"

## How to run Offload Advisor on Linux

See the Advisor Cookbook here: https://software.intel.com/en-us/advisor-cookbook

## How to run Vtune GPU Hotspot on Linux
    vtune -collect gpu-hotspots -target-duration-type=veryshort -allow-multiple-runs -app-working-dir . -- make run

## How to run Vtune GPU Hotspot on Windows
    vtune -collect gpu-hotspots -target-duration-type=veryshort -allow-multiple-runs -app-working-dir . -- x64\Release\nbody.exe

## Example of Output
    ===============================
     Initialize Gravity Simulation
     Target Device: Intel(R) Gen9
     nPart = 16000; nSteps = 10; dt = 0.1
    ------------------------------------------------
     s       dt      kenergy     time (s)    GFlops
    ------------------------------------------------
     1       0.1     26.405      0.28029     26.488
     2       0.2     313.77      0.066867    111.03
     3       0.3     926.56      0.065832    112.78
     4       0.4     1866.4      0.066153    112.23
     5       0.5     3135.6      0.065607    113.16
     6       0.6     4737.6      0.066544    111.57
     7       0.7     6676.6      0.066403    111.81
     8       0.8     8957.7      0.066365    111.87
     9       0.9     11587       0.066617    111.45
     10      1       14572       0.06637     111.86
    
    # Total Time (s)     : 0.87714
    # Average Performance : 112.09 +- 0.56002
    ===============================
    Built target run
