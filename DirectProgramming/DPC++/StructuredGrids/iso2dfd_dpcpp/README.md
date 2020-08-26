# ISO2DFD sample

ISO2DFD: Intel® oneAPI DPC++ Language Basics Using 
2D-Finite-Difference-Wave Propagation

The ISO2DFD sample refers to Two-Dimensional Finite-Difference Wave Propagation in Isotropic Media.  It is a two-dimensional stencil to simulate a wave propagating in a 2D isotropic medium and illustrates the basics of the DPC++ programming language using direct programming.

A complete code walk-through for this sample can be found at:
https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd

For comprehensive instructions regarding DPC++ Programming, go to
https://software.intel.com/en-us/oneapi-programming-guide
and search based on relevant terms noted in the comments.

  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to offload the computation to GPU using Intel® oneAPI DPC++/C++ Compiler
| Time to complete                  | 10 minutes


## Purpose

ISO2DFD is a finite difference stencil kernel for solving the 2D acoustic isotropic wave equation.  In 
this sample, we chose the problem of solving a Partial Differential Equation (PDE), using a 
finite-difference method, to illustrate the essential elements of the DPC++ programming language: 
queues, buffers/accessors, and kernels. Use it as an entry point to start programming in DPC++ or as a 
proxy to develop or better understand complicated code for similar problems. 

Using Data Parallel C++, the sample will explicitly run on the GPU as well as CPU to calculate a 
result. The output will include GPU device name. The results from the two devices are compared and, if 
the sample ran correctly, report a success message. The output of the wavefield can be plotted using 
the SU Seismic processing library, which  has utilities to display seismic wavefields and can be 
downloaded from John Stockwell’s SeisUnix GitHub* (https://github.com/JohnWStockwellJr/SeisUnix/wiki/
Seismic-Unix-install-on-Ubuntu)


## Key implementation details

SYCL implementation explained.  

* DPC++ queues (including device selectors and exception handlers).
* DPC++ buffers and accessors.  
* The ability to call a function inside a kernel definition and pass accessor arguments as pointers. A 
function called inside the kernel performs a computation (it updates a grid point specified by the 
global ID variable) for a single time step.  


## License

This code sample is licensed under MIT license.  

##  Building the `iso2dfd` Program for CPU and GPU

### Include Files 

The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, 
FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI 
Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands. 

   ```    
    cd iso2dfd_dpcpp &&  
    mkdir build &&  
    cd build &&  
    cmake .. &&  
    make -j 
    ```

2. Run the program on Gen9 

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

## Running the Sample
### Application Parameters 

You can  execute the code with different parameters. For example the following command will run the iso2dfd executable using a 1000x1000 grid size and it will iterate over 2000 time steps.

    ```
    ./iso2dfd 1000 1000 2000
    ```	
 	
   Usage: ./iso2dfd n1 n2 Iterations

	 n1 n2      : Grid sizes for the stencil
	 Iterations : Number of timesteps.

   * Find graphical output for sample execution in the online tutorial at:
     https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd

### Example of Output 

    ```
    Initializing ... 
    Grid Sizes: 1000 1000
    Iterations: 2000

    Computing wavefield in device ..
    Running on Intel(R) Gen9 HD Graphics NEO
    The Device Max Work Group Size is : 256
    The Device Max EUCount is : 24
    SYCL time: 3282 ms

    Computing wavefield in CPU ..
    Initializing ... 
    CPU time: 8846 ms

    Final wavefields from device and CPU are equivalent: Success
    Final wavefields (from device and CPU) written to disk
    Finished.  
    [100%] Built target run
    ```
