# Multiple Simple Random Sampling without replacement

Multiple Simple Random Sampling without replacement shows how to use the oneMKL library's random number generation (RNG) functionality to generate K>>1 simple random length-M samples without replacement from a population of size N (1 ≤ M ≤ N).

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04; Windows 10
| Hardware            | Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL beta
| What you will learn | How to use oneMKL's random number generation functionality
| Time to complete    | 15 minutes


## Purpose

The sample demonstrates Partial Fisher-Yates Shuffle algorithm conducts 11 969 664 experiments. Each experiment, which generates a sequence of M unique random natural numbers from 1 to N, is actually a partial length-M random shuffle of the whole population of N elements. Because the main loop of the algorithm works as a real lottery, each experiment is called "lottery M of N" in the program.
The program uses M=6 and N=49, stores result samples (sequences of length M) in a single array.

This sample uses oneMKL's random number generation functionality to produce the random numbers. oneMKL RNG has APIs that can be called from the host, and APIs that can be called from within a kernel; both kinds of APIs are illustrated.

This sample performs its computations on the default DPC++ device. You can set the `SYCL_DEVICE_TYPE` environment variable to `cpu` or `gpu` to select the device to use.


## Key Implementation Details

This sample illustrates how to create an RNG engine object (the source of pseudo-randomness), a distribution object (specifying the desired probability distribution), and finally generate the random numbers themselves. Random number generation can be done from the host, storing the results in a DPC++ buffer or USM pointer, or directly in a DPC++ kernel.

In this sample, a Philox 4x32x10 generator is used, and a uniform distribution is the basis for the algorithm. oneMKL provides many other generators and distributions to suit a range of applications.


## License

This code sample is licensed under the MIT license.


## Building the Multiple Simple Random Sampling without replacement Sample

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Run `make` to build and run the sample. Three programs are generated, which illustrate different APIs for random number generation.

You can remove all generated files with `make clean`.

### On a Windows* System
Run `nmake` to build and run the sample. `nmake clean` removes temporary files.

## Running the Multiple Simple Random Sampling without replacement Sample

### Example of Output
If everything is working correctly, after building you will see step-by-step output from each of the three example programs, providing the results of lottery.
```
./lottery

Multiple Simple Random Sampling without replacement
Unified Shared Memory Api
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,

./lottery_usm

Multiple Simple Random Sampling without replacement
Unified Shared Memory Api
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Results with Host API:
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,

./lottery_device_api

Multiple Simple Random Sampling without replacement
Device Api
---------------------------------------------------
M = 6, N = 49, Number of experiments = 11969664
Sample 11969661 of lottery of 11969664: 19, 5, 17, 27, 44, 34,
Sample 11969662 of lottery of 11969664: 31, 39, 6, 19, 48, 15,
Sample 11969663 of lottery of 11969664: 24, 11, 29, 44, 2, 20,
```
