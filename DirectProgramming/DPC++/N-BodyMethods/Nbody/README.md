# Nbody sample
An N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity. This nbody sample code is implemented using C++ and DPC++ language for Intel CPU and GPU. 
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler beta;
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes

## Purpose
Nbody sample code simulates 16000 particles and for 10 integration steps. Each particles position, velocity and acceleration parameters are dependent on other (N-1) particles. This algorithm is highly data parallel and a perfect candidate to offload to GPU. The code demonstartes how to deal with multiple device kernels which can be enqueued into a DPC++ queue for execution and how to handle parallel reductions.

## Key Implementation Details 
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## License  
This code sample is licensed under MIT license. 

## Building the Program for CPU and GPU

### Include Files  
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands. 
``` 
$ mkdir build
$ cd build
$ cmake ..
$ make
```
2. Run the program 
    ```
    make run
    ```

3. Clean the program 
    ```
    make clean
    ```
### On a Windows* System Using Visual Studio* Version 2017 or Newer
* Build the program using VS2017 or VS2019
      Right click on the solution file and open using either VS2017 or VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.

* Build the program using MSBuild
      Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
      Run - MSBuild Nbody.sln /t:Rebuild /p:Configuration="Release"

### Application Parameters 
You can modify the NBody simulation parameters from within GSimulation.cpp. The configurable parameters include:
  set_npart(__);
  set_nsteps(__);
  set_tstep(__);
  set_sfreq(__);
Below are the default parameters:
Number of particles (npart) is 16000
Number of integration steps (nsteps) is 10
Time delta (tstep) is 0.1s
Sample frequency (sfreq) is 1

## Example of Output
    ===============================
     Initialize Gravity Simulation
     Target Device: Intel(R) Gen9
     nPart = 16000; nSteps = 10; dt = 0.1
    ------------------------------------------------
     s       dt      kenergy     time (s)    GFLOPS
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
