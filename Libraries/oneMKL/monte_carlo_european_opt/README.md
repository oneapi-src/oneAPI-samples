# Monte Carlo European Options Sample

Monte Carlo European Options shows how to use the oneMKL library's random number generation (RNG) functionality to compute European option prices.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL beta
| What you will learn | How to use oneMKL's random number generation functionality
| Time to complete    | 15 minutes


## Purpose

This sample uses a Monte Carlo (randomized) method to estimate European call and put option values, based on a stochastic model of stock prices. A large number of possible realizations of the stock price over time are generated, and an estimation of the call and put options is made by averaging their values in each realization.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a DPC++ buffer or USM pointer, or directly in a DPC++ kernel.

In this sample, a Philox 4x32x10 generator is used, and a log-normal distribution is the basis for the Monte Carlo simulation. oneMKL provides many other generators and distributions to suit a range of applications. After generating the random number input for the simulation, prices are calculated, and then averaged using DPC++ reduction functions.

Two versions of the sample are provided: one using DPC++ buffer objects, and one using raw pointers (Unified Shared Memory).


## License

This code sample is licensed under the MIT license.

## Building the Monte Carlo European Options Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample. Two programs will be generated: a buffer version (`mc_european`) and a USM version (`mc_european_usm`).

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample programs. `nmake clean` removes temporary files.


## Running the Monte Carlo European Options Sample
If everything is working correctly, both buffer and USM versions of the program will run the Monte Carlo simulation. After simulation, results will be checked against the known true values given by the Black-Scholes formula, and the absolute error is output.

```
./mc_european

Monte Carlo European Option Pricing Simulation
Buffer Api
----------------------------------------------
Number of options = 2048
Number of samples = 262144
put_abs_err  = 0.0372205
call_abs_err = 0.0665814

./mc_european_usm

Monte Carlo European Option Pricing Simulation
Unified Shared Memory Api
----------------------------------------------
Number of options = 2048
Number of samples = 262144
put_abs_err  = 0.0372205
call_abs_err = 0.0665814
```
