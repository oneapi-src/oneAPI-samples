# Water Molecule Diffusion Sample
motionsim: Intel® oneAPI DPC++ Language Basics Using a Monte Carlo Simulation

This code sample implements a simple example of a Monte Carlo simulation of the diffusion of water molecules in tissue. This kind of computational experiment can be used to simulate acquisition of a diffusion signal for dMRI.

The model for the simulation consists of water molecules moving through a 2D array of cells in a tissue sample (water molecule diffusion). In this code sample, we use a uniform rectilinear 2D array of digital cells, where cells are spaced regularly along each direction and are represented by circles.

Water molecule diffusion is simulated by defining a number of particles P (simulated water molecules) at random positions in the grid, followed by random walks of these particles in the ensemble of cells in the grid. During the random walks, particles can move randomly inside or outside simulated cells. The positions of these particles at every time step in the simulation, the number of times they go through a cell membrane (in/out), as well as the time every particle spends inside and outside cells can be recorded. These measurements are a simple example of useful information that can be used to simulate an MR signal. 

For comprehensive instructions regarding DPC++ Programming, go to
https://software.intel.com/en-us/oneapi-programming-guide
and search based on relevant terms noted in the comments.

 For more information and details: https://software.intel.com/en-us/articles/vectorization-of-monte-carlo-simulation-for-diffusion-weighted-imaging-on-intel-xeon
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10 or Windows Server 2017
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel Data Parallel C++ Compiler (beta)
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 20 minutes

Performance number tabulation [if applicable]

| motionsim sample                  | Performance data
|:---                               |:---
| Scalar baseline -O2               | 1.0
| SYCL                              | 
| OpenMP offload                    | 
 
## Key Implementation Details

SYCL implementation explained. 

* DPC++ queues (including device selectors and exception handlers).
* DPC++ buffers and accessors.  
* The ability to call a function inside a kernel definition and pass accessor arguments as pointers.
* Optimization using API-based programming and Atomic Functions.

SYCL implementation explained in further detail in source code.
## How Other Tools (Intel Libraries or Intel Tools) are used
Intel® Math Kernel Library (MKL) is used for random number generation on the cpu and device. Precise generators are used within this library to ensure that the numbers generated on the cpu and device are relatively equivalent (relative accurracy 10E-07).  
## License
This code sample is licensed under MIT license. Please see the `License.txt` file for more information.  
## Building the `particle_diffusion` Program for CPU and GPU

### Include Files  
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your 
development system".  

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, 
FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI 
Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

## Build and run

### On a Windows\* System Using Microsoft Visual Studio 2017 or Newer

#### Build the motionsim Program Using Visual Studio 2017 or Visual Studio 2019

##### 1. Right click on the solution file (.sln) and open it using either Visual Studio 2017 or Visual Studio 2019
##### 2. From Visual Studio, right click on the project solution file in solution explorer and select rebuild
##### 3. From top menu select Debug -> Start Without Debugging

#### Build the motionsim Program Using MSBuild

##### 1. Open "x64 Native Tools Command Prompt for VS 2017" or "x64 Native Tools Command Prompt for VS 2019" as Administrator (right click application and select Run as Administrator)
##### 2. Build
From the particle diffusion Project directory:  

    > MSBuild Particle_Diffusion.sln /t:Rebuild /p:Configuration="Release"

### On a Linux\* System Using CMake

#### 1. Enter Particle Diffusion Directory
    $ cd particle_diffusion
#### 2. Build motionsim Program Using CMake
    $ mkdir build && cd build && cmake .. && make -s -j
#### 3. Run
##### 3a. Run Using make (Default Parameters)
    $ make run
##### 3b. Run Using Binary File (Custom Parameters)
The following table describes each command line parameter (applies to Linux\* based builds only)

|    Flag and Argument          |    Description               |    Range of Possible Values    |    Default    
|:---                           |:---                          |:---                            |:---    
| `-i num_iterations`           | Number of iterations         | [1, &#8734;]                   | 10000    
| `-p num_particles`            | Number of particles          | [1, &#8734;]                   | 256    
| `-g grid_size`                | Size of square grid          | [1, &#8734;]                   | 22
| `-r rng_seed`                 | Random number generator seed | [-&#8734;, &#8734;]            | 777    
| `-c cpu_flag`                 | Turns cpu comparison on/off  | [1 \| 0]                       | 0    
| `-o output_flag`              | Turns grid output on/off     | [1 \| 0]                       | 1    
| `-h`                          | Help message.                |                                |    

You can run the program using the above parameters with the application binary:  

    $ ./src/motionsim.exe

Example usage:  

    $ ./src/motionsim.exe -i 1000 -p 200 -g 30 -r 777 -c 1 -o 0

Note:

* If the grid size specified is greater than 44, the application will not print the grid even if the grid output flag is on

* If a particular parameter is not specified, the application will choose the default value for that parameter

* Typing `$ ./src/motionsim.exe -h` displays a brief help message and exits the program

#### 4. Clean up
    $ cd .. && rm -r build
## Example Execution (Linux\* System)

    $ make run

    Scanning dependencies of target run

    **Running with default parameters**

    Running on: Intel(R) Gen9
    Device Max Work Group Size: 256
    Device Max EUCount: 24
    Number of iterations: 10000
    Number of particles: 256
    Size of the grid: 22
    Random number seed: 777

    Device Offload time: 0.391211 s


    **********************************************************
    *                           DEVICE                       *
    **********************************************************

    ********************** FULL GRID: **********************

    0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0  15   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0  96   0 114   0   0  25   2   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0 113 358 197  17 224 286   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   3  47   0  72 725  53   0   0   0   0   0   0   0   0
    0   0   0   0   0   3   0   0  32 266 780 684 482 861 251   0   0   0   0   0   0   0
    0   0   0   0   0 165   0 276 1076 705 943 1457 1700 1628 1712   3   0   0   0   0   0   0
    0   0   0   0   0 185 771 1801 942 1203 2774 2576 3167 1490 582 246  19   0   0   0   0   0
    0   0   0   0   0   1 790 1852 2305 2599 4127 7126 3349 4056 2867 493 198  71   0   0   0   0
    0   0   0   0   0   0 940 870 4171 6239 10529 10297 9057 6166 4439 2097 1562 490   0   0   0   0
    0   0   0   0   0   0 224 745 4506 6387 18315 24694 18479 11373 5567 2003 1748 601 146   0   0   0
    0   0   0   0   3 119 1234 2316 5819 9242 22970 43726 15391 9562 5574 1958 1693 1099 242   0   0   0
    0   0   0   0 178 502 1388 1627 6685 10219 16732 18655 9476 5441 4427 2885 1667 814 441 161   0   0
    0   0   0   0  88 325 1210 2022 3891 8123 6318 6835 6728 4048 3234 2532 725  70 506  49   0   0
    0   0   0  42 292   0 519 1723 3113 3787 3504 5828 3916 2644 2141 1471 126 102   0   0   0   0
    0   0   0   0   0   0   0 653 949 716 1707 1566 1669 290 1053 627 118 218   0   0   0   0
    0   0   0   0   0 119 118 207 250 431 273 1112 537 137 193 152  14 450   0   0   0   0
    0   0   0   0   0 383 174 163 187  22  67 822 195   4  21   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0  97   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0 172 131   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0

    ***************** FINAL SNAPSHOT: *****************

    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   1   0   0   0   0   1   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   1   0   0   0   0   3   1   0   0   1   0   0   0   0
    0   0   0   0   0   0   0   0   0   1   0   2   1   0   0   0   0   1   0   0   0   0
    0   0   0   0   0   0   0   1   1   1   0   0   2   3   0   0   2   1   0   0   0   0
    0   0   0   0   0   0   1   0   0   1   1   0   0   0   0   0   0   1   0   0   0   0
    0   0   0   0   0   0   2   0   1   1   2   0   0   1   0   0   0   0   0   0   0   0
    0   0   0   0   0   1   0   0   0   0   1   2   1   1   2   1   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   1   1   1   0   1   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   1   0   0   1   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0

    ************* NUMBER OF PARTICLE ENTRIES: *************

    0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   9   0   9   0   0   9   2   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0  12  39  42   2   5  25   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   1  15   0  13  76   9   0   0   0   0   0   0   0   0
    0   0   0   0   0   3   0   0   6  33  84  67  51 123  38   0   0   0   0   0   0   0
    0   0   0   0   0  23   0  24  79  75 118 154 178 131 153   2   0   0   0   0   0   0
    0   0   0   0   0  26  55 183 124 114 220 338 337 143  63  42   1   0   0   0   0   0
    0   0   0   0   0   1  58 185 257 258 479 801 405 462 312  64  19   9   0   0   0   0
    0   0   0   0   0   0  80 111 431 668 1044 1141 988 711 458 217 128  34   0   0   0   0
    0   0   0   0   0   0  17 110 530 814 1951 2745 1851 1291 643 225 198  65  13   0   0   0
    0   0   0   0   1  15 105 209 649 1018 2426 4770 1620 1030 599 181 190 114  34   0   0   0
    0   0   0   0  26  37 160 203 645 1083 1624 1905 965 613 509 324 176  86  31  13   0   0
    0   0   0   0   6  52 119 169 447 825 789 762 755 417 324 273  91  11  33   7   0   0
    0   0   0   8  27   0  47 156 303 414 444 596 370 316 172 152  17  11   0   0   0   0
    0   0   0   0   0   0   0  76 111  91 210 181 171  33  88  65  17  19   0   0   0   0
    0   0   0   0   0   3   3  20  34  35  41 125  59  28  26  21   8  27   0   0   0   0
    0   0   0   0   0  31  14  23   8   2   5  70  16   1   3   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0  19   9   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    **********************************************************
    *                        END DEVICE                      *
    **********************************************************


    Built target run
    $
