# Black-Scholes Sample

Black-Scholes shows how to use oneMKL's Vector Math (VM) and Random Number Generator (RNG) functionality to calculate the prices of options using the Black-Scholes formula for suitable randomly-generated portfolios.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL beta
| What you will learn | How to use oneMKL's vector math and random number generation functionality
| Time to complete    | 15 minutes


## Purpose

The Black-Scholes formula is widely used in financial markets as a basic prognostic tool. The ability to calculate it quickly for a large number of options has become a neccesity and represents a classic problem in parallel computation.

The sample generates first a portfolio within given constraints using a uniform distribution and a Philox-type generator provided by oneMKL's RNG API.
Examples of both host-based APIs and device-based APIs for random number generation are provided.

Then, the Black-Scholes formula is used in two distinct implementations:
    * scalar kernel containing SYCL scalar functions, which will be parallelized by the DPC++ compiler
    * vector-based implementation, which uses oneMKL vector math functionality in sequential calls to evaluate the transcendental functions used in the formula.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to use oneMKL VM to implement explicit vectorization of calculations over vectors of large size. The key point illustrated is the ability to asynchronously mix kernels with VM function calls and their results. When using USM, the user is in charge of noting explicit dependencies between calls -- this essential technique is also illustrated.

This sample also illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a DPC++ buffer or USM pointer, or directly in a DPC++ kernel using device API.

In this sample, a Philox 4x32x10 generator is used. It is a lightweight counter-based RNG well-suited for parallel computing.


## License

This code sample is licensed under the MIT license.

## Building the Black-Scholes Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample. One program is generated which will use four different interfaces in turn.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.


## Running the Black-Scholes Sample
If everything is working correctly, the program will exercise different combinations of APIs for both single and double precision (if available), and print a summary of its computations.

```
running on:
       device name: Intel(R) Gen9
    driver version: 0.91

running floating-point type float


running USM mkl::rng
    <s0> = 29.9999
    <x> = 29.9994
    <t> = 1.50002
running USM mkl::vm
    <opt_call> = 27.148
    <opt_put>  = 4.12405
```
...
```
running Buffer mkl::rng_device
    <s0> = 30.2216
    <x> = 30.0563
    <t> = 1.49756
running Buffer dpcpp
    <opt_call> = 27.3496
    <opt_put>  = 4.10354
```
