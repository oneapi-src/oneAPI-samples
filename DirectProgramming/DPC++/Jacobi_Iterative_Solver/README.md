# `Jacobi Iterative Solver` Sample

The `Jacobi Iterative Solver` Sample shows how to use the Intel Base Toolkit to use CPU, GPU and multi Gpu offload using SYCL and the differences in sample runtime using different targes.  

The `Jacobi Iterative Solver` Sample refers to a system of equations represented by two input matrices with the first one being the number of unknown variables ant the second one being results. It calculates the results using the Jacobi Iterative method and compares the newly calculated results with the old ones. 

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | IntelÂ® oneAPI DPC++/C++ Compiler
| What you will learn               | How to different targets impact the behaviour of the sample, how does the oneAPI random number generators work. 
| Time to complete                  | 10 minutes


## Purpose

This sample starts with a CPU oriented application and shows how to use SYCL and the oneAPI tools to offload regions of the code to the target system's GPU.  We'll use Intel Advisor to conduct offload modeling to identify code regions that will benefit the most from GPU offload. Once the initial offload is complete, we'll walk through how to develop an optimization strategy by iteratively optimizing the code baed on opportunities exposed Intel Advisor to run roofline analysis. 

ISO3DFD is a finite difference stencil kernel for solving the 3D acoustic isotropic wave equation, which can be used as a proxy for propagating a seismic wave. In this sample, kernels are implemented as 16th order in space, with symmetric coefficients, and 2nd order in time scheme without boundary conditions.

## Key Implementation Details

The basic DPC++ implementation explained in the code includes the use of the following :
* DPC++ local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each DPC++ workgroup)
* Code for Shared Local Memory (SLM) optimizations
* DPC++ kernels (including parallel_for function and range<1> objects)


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `Jacobi Iterative Solver` Program for CPU and GPU


> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html.

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands.
```
$ mkdir build
$ cd build
$ cmake ..
```

2. Run the program :
```
$ make run_cpu 
```
> Note: the following run commands area also available and correspond to the specific build targets. 

    make run_cpu
    make run_gpu

> Note: the command below will be available when multi GPU enviroment can bu run on devcloud

    make run_multi_gpu

3. Clean the program using:

```
$ make clean
```

4. Example of Output
```
[ 16%] Building CXX object src/jacobi_gpu/CMakeFiles/jacobi_gpu_iterative_solver.dir/jacobi_gpu_iterative_solver.cpp.o
[ 33%] Linking CXX executable jacobi_gpu_iterative_solver
[ 33%] Built target jacobi_gpu_iterative_solver
[ 50%] Building CXX object src/jacobi_cpu/CMakeFiles/jacobi_cpu_iterative_solver.dir/jacobi_cpu_iterative_solver.cpp.o
[ 66%] Linking CXX executable jacobi_cpu_iterative_solver
[ 66%] Built target jacobi_cpu_iterative_solver
[ 83%] Building CXX object src/jacobi_multi_gpu/CMakeFiles/jacobi_multi_gpu_iterative_solver.dir/jacobi_multi_gpu_iterative_solver.cpp.o
[100%] Linking CXX executable jacobi_multi_gpu_iterative_solver
[100%] Built target jacobi_multi_gpu_iterative_solver

Scanning dependencies of target run_cpu
./jacobi_cpu_iterative_solver
Device : Intel(R) Core(TM) i7-10610U CPU @ 1.80GHz

Matrix generated, time elapsed: 0.31112 seconds.
[5122.01 263.22 1.67 626.22 317 -333.22 947.8 -852.83 -808.99 ][277.63]
[277.63 -4634.73 529.95 -657.22 -564.18 601.12 676.36 452.62 314.03 ][740.29]
[740.29 499.7 -4774.05 794.3 -156.33 -237.9 397.63 -160.86 916.96 ][76.47]
[76.47 -693.42 -135.63 4075.95 -287.56 993 936.34 -28.63 86.55 ][833.52]
[833.52 769.69 -400.31 443.35 -5587.83 431.49 453.66 556.13 845.82 ][844.33]
[844.33 549.12 440.36 416.48 -236.84 -4121.63 -460.53 -236.51 -359.54 ][-358.82]
[-358.82 -655.35 -569.26 -982.24 102.27 522.64 -4836.28 -534.2 -552.49 ][545.54]
[545.54 -403.51 249.11 918.1 -575.88 -151.27 159.93 3964.62 -770 ][-216.26]
[-216.26 751.31 267.88 691.34 -161.82 973.27 908.53 -175.76 -4150.42 ][-813.02]

Computations complete, time elapsed: 0.295054 seconds.
Total number of sweeps: 30
Checking results
All values are correct.

Check complete, time elapsed: 0.00481672 seconds.
Total runtime is 0.750036 seconds.
X1 equals: 0.09895873327
X2 equals: -0.15802401129
X3 equals: 0.03854686450
X4 equals: 0.16617806412
X5 equals: -0.12925305745
X6 equals: 0.11829640135
X7 equals: -0.13914309700
X8 equals: -0.09521910620
X9 equals: 0.19864875400
Built target run_cpu
```

