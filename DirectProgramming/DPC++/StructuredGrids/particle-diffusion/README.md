# water molecule diffusion sample

motionsim: Intel® oneAPI DPC++ Language Basics Using a Monte Carlo Simulation

This code sample implements a simple example of a Monte Carlo simulation of the diffusion of water molecules in tissue. IT reflects basid DPC++ programming as well as some techniques for optimization (API-based programming and Atomic Functions).

For comprehensive instructions regarding DPC++ Programming, go to
https://software.intel.com/en-us/oneapi-programming-guide
and search based on relevant terms noted in the comments.

 For more information and details: https://software.intel.com/en-us/articles/vectorization-of-monte-carlo-simulation-for-diffusion-weighted-imaging-on-intel-xeon
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10 or Windows Server 2017
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel Data Parallel C++ Compiler (beta)
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes

Performance number tabulation [if applicable]

| motionsim sample                      | Performance data
|:---                               |:---
| Scalar baseline -O2               | 1.0
| SYCL                              | 
| OpenMP offload                    | 

## Purpose

Using the Monte Carlo simulation, the Particle Diffusion sample provides simulation of the 
diffusion of water molecules in tissue  This kind of computational experiment can be used to 
simulate acquisition of a diffusion signal for dMRI.

The model for the simulation consists of water molecules moving through a 2D array of cells in a 
tissue sample (water molecule diffusion). In this code sample, we use a uniform rectilinear 2D 
array of digital cells, where cells are spaced regularly along each direction and are represented 
by circles.

Water molecule diffusion is simulated by defining a number of particles P (simulated water 
molecules) at random positions in the grid, followed by random walks of these particles in the 
ensemble of cells in the grid. During the random walks, particles can move randomly inside or 
outside simulated cells. The positions of these particles at every time step in the simulation, 
the number of times they go through a cell membrane (in/out), as well as the time every particle 
spends inside and outside cells can be recorded. These measurements are a simple example of 
useful information that can be used to simulate an MR signal. 

The Particle Diffusion sample is intended to show the basic elements of the DPC++ programming 
language as well as some basic optimizations as generating random numbers in the device (using 
functionality from the oneAPI oneMKL library), as well as atomic functions to prevent memory 
access inconsistencies. 


## Key implementation details 

SYCL implementation explained. 

* DPC++ queues (including device selectors and exception handlers).
* DPC++ buffers and accessors.  
* The ability to call a function inside a kernel definition and pass accessor arguments as pointers.
* Optimization using API-based programming and Atomic Functions.


## License  

This code sample is licensed under MIT license.  


## Building the `Particle_Diffusion` Program for CPU and GPU

### Include Files  
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your 
development system".  

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, 
FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI 
Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:
1. Build the motionsim program using the following `cmake` commands. 
    ```
    cd Particle_Diffusion &&  
    mkdir build &&  
    cd build &&  
    cmake ../. &&  
    make VERBOSE=1  
    ```

2. Run the program:
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```

### on Windows - Visual Studio 2017 or newer
   * Build the program using VS2017 or VS2019  
    Right click on the solution file and open using either VS2017 or VS2019 IDE  
    Right click on the project in Solution explorer and select Rebuild  
    From top menu select Debug -> Start without Debugging  

## Running the Sample

### Application Parameters 
   *  You can  execute the code with different parameters. For example, the following command will run the motionsim executable using 777 as the seed for the random number generator, and will iterate over 10000 time steps 

    ```
    ./src/motionsim.exe 10000 777
    ```
    Usage: src/motionsim.exe <Number of Iterations>  <Seed for RNG>


### Example of Output
The output grid show the cells indicating the number of times that particles have been inside the cell and how the number of particles diffuses from the center of the grid.

    ```
    type src/motionsim.exe 10000 777 (or type 'make run')

    $ src/motionsim.exe 10000 777

    Running on:: Intel(R) Gen9 HD Graphics NEO
    The Device Max Work Group Size is : 256
    The Device Max EUCount is : 72
    The number of iterations is : 10000
    The number of particles is : 20

    Offload: Time: 137


    ********************** OUTPUT GRID:

      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0 367  27  16   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0  84 750  98  84   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0 669   0 116  55   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0 130 211 250 170  35  30 261   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0  10 353 539 243 809  61 878 174   0   0   0   0   0   0  
      0   0   0   0   0   0   1 118 1628 1050 1678 887 864 272  80 390   0   0   0   0   0  
      0   0   0   0   0   0   0  39 1173 1660 3549 1263 1155 2185 234   0   0   0   0   0   0  
      0   0   0   0   0   0 306 599 609 537 550 1134 1172 1261  13   0   0   0   0   0   0  
      0   0   0   0   0   0 283 120  92 282 851 512 658 872  40   0   0   0   0   0   0  
      0   0   0   0   0 157 284 133 817 151 175 271 147 286  57   0   0   0   0   0   0  
      0   0   0   0   0   0 294 428   0   0   0   0   0  17   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0 364   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0 182   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  
      ```
$

