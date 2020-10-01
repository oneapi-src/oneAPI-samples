# Monte Carlo Pi Estimation Sample

Monte Carlo Pi Estimation shows how to use the oneMKL library's random number generation (RNG) functionality to estimate the value of &pi;.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL beta
| What you will learn | How to use oneMKL's random number generation functionality
| Time to complete    | 15 minutes


## Purpose

Monte Carlo Pi Estimation uses a randomized method to estimate the value of &pi;. The basic idea is to generate random points within a square, and to check what fraction of these random points lie in a quarter-circle inscribed in that square. The expected value is the ratio of the areas of the quarter-circle and the square, namely &pi;/4, so we can take the observed fraction of points in the quarter-circle as an estimate of &pi;/4.

This sample uses oneMKL's random number generation functionality to produce the random points. oneMKL RNG has APIs that can be called from the host, and APIs that can be called from within a kernel; both kinds of APIs are illustrated.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a DPC++ buffer or USM pointer, or directly in a DPC++ kernel.

In this sample, a Philox 4x32x10 generator is used, and a uniform distribution is the basis for the Monte Carlo simulation. oneMKL provides many other generators and distributions to suit a range of applications.


## License

This code sample is licensed under the MIT license.


## Building the Monte Carlo Pi Estimation Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample. Three programs are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

## Running the Monte Carlo Pi Estimation Sample

### Example of Output
If everything is working correctly, after building you will see step-by-step output from each of the three example programs, providing the generated estimate of &pi;.
```
./mc_pi

Monte Carlo pi Calculation Simulation
Buffer API
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14139
Exact value of Pi = 3.14159
Absolute error = 0.000207487

./mc_pi_usm

Monte Carlo pi Calculation Simulation
Unified Shared Memory API
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14139
Exact value of Pi = 3.14159
Absolute error = 0.000207487

./mc_pi_device_api

Monte Carlo pi Calculation Simulation
Device API
-------------------------------------
Number of points = 120000000
Estimated value of Pi = 3.14139
Exact value of Pi = 3.14159
Absolute error = 0.000207487
```
