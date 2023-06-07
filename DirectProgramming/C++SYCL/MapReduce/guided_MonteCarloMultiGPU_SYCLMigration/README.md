# `MonteCarloMultiGPU` Sample

The `MonteCarloMultiGPU` sample evaluates fair call price for a given set of European options using the Monte Carlo approach. MonteCarlo simulation is one of the most important algorithms in quantitative finance. This sample uses a single CPU Thread to control multiple GPUs.

> **Note**: This sample is migrated from NVIDIA CUDA sample. See the [MonteCarloMultiGPU](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/MonteCarloMultiGPU) sample in the NVIDIA/cuda-samples GitHub.

| Property                       | Description
|:---                               |:---
| What you will learn               | How to begin migrating CUDA to SYCL
| Time to complete                  | 15 minutes


## Purpose
MonteCarlo method is basically a way to compute expected values by generating random scenarios and then averaging them, it is actually very efficient to parallelize. With the GPU we can reduce this problem by parallelizing the paths. That is, we can assign each path to a single thread, simulating thousands of them in parallel, with massive savings in computational power and time.

This sample contains four versions in the following folders:

| Folder Name                          | Description
|:---                                  |:---
| 01_sycl_dpct_output                  | Contains output of Intel® DPC++ Compatibility Tool used to migrate SYCL-compliant code from CUDA code, this SYCL code has some unmigrated code which has to be manually fixed to get full functionality, the code does not functionally work.
| 02_sycl_dpct_migrated                | Contains Intel® DPC++ Compatibility Tool migrated SYCL code from CUDA code with manual changes done to fix the unmigrated code to work functionally.
| 03_sycl_migrated                     | Contains manually migrated SYCL code from CUDA code.
| 04_sycl_migrated_optimized           | Contains manually migrated SYCL code from CUDA code with performance optimizations applied.

## Prerequisites
| Property                       | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler

## Key Implementation Details

Pricing of European Options can be done by applying the Black-Scholes formula and with MonteCarlo approach.

MonteCarlo Method first generates a random number based on a probability distribution. The random number then uses the additional inputs of volatility and time to expiration to generate a stock price. The generated stock price at the time of expiration is then used to calculate the value of the option. The model then calculates results over and over, each time using a different set of random values from the probability functions

The first stage of the computation is the generation of a normally distributed N(0, 1)number sequence, which comes down to uniformly distributed sequence generation.Once we’ve generated the desired number of samples, we use them to compute an expected value and confidence width for the underlying option. 

The Black-Scholes model relies on fixed inputs (current stock price, strike price, time until expiration, volatility, risk free rates, and dividend yield).The model is based on geometric Brownian motion with constant drift and volatility.We can calculate the price of the European put and call options explicitly using the Black–Scholes formula.

After repeatedly computing appropriate averages, the estimated price of options can be obtained, which is consistent with the analytical results from Black-Scholes model.

>**Note**: This sample application demonstrates the CUDA MonteCarloMultiGPU using key concepts such as Random Number Generator and Computational Finance.

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `MonteCarloMultiGPU` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
>For more information on environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On Linux*
Perform the following steps:
1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

    By default, this command sequence will build the `03_sycl_migrated`, and `04_sycl_migrated_optimized` versions of the program.
    
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `MonteCarloMultiGPU` Sample

### On Linux

You can run the programs for CPU and GPU. The commands indicate the device target.
    
1. Run `03_sycl_migrated` for CPU and GPU.
    ```
    make run_cpu
    make run_gpu (runs on Level-Zero Backend)
    make run_gpu_opencl (runs on OpenCL Backend)
    ```
    
2. Run `04_sycl_migrated_optimized` for CPU and GPU.
    ```
    make run_smo_cpu
    make run_smo_gpu (runs on Level-Zero Backend)
    make run_smo_gpu_opencl (runs on OpenCL Backend)
    ```

### Run the `MonteCarloMultiGPU` Sample in Intel&reg; DevCloud

When running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)

You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`.

1. Open a terminal on a Linux* system.
2. Log in to Intel&reg; DevCloud.
    ```
    ssh devcloud
    ```
3. Download the samples.
    ```
    git clone https://github.com/oneapi-src/oneAPI-samples.git
    ```
4. Change to the `guided_MonteCarloMultiGPU_SYCLMigration` directory.
    ```
    cd ~/oneAPI-samples/DirectProgramming/DPC++/MapReduce/guided_MonteCarloMultiGPU_SYCLMigration
    ```
5. Configure the sample for a GPU node using `qsub`and choose the backend needed either OpenCL or Level Zero.
    ```
    qsub  -I  -l nodes=1:gpu:ppn=2 -d .
    ```
    - `-I` (upper case I) requests an interactive session.
    - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node.
    - `-d .` makes the current folder as the working directory for the task.
6. Perform build steps as you would on Linux.
7. Run the sample.
8. Clean up the project files.
    ```
    make clean
    ```
9. Disconnect from the Intel&reg; DevCloud.
    ```
    exit
    ```

### Example Output
The following example is for `03_sycl_migrated` for GPU on Intel(R) UHD Graphics P630 [0x3e96] with Level Zero Backend.
```
./a.out Starting...

MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of Devices       = 1
Total number of options = 12
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): Device statistics, streamed
Device #0: Intel(R) UHD Graphics P630 [0x3e96]
Options         : 12
Simulation paths: 262144

Total time (ms.): 11.149000
        Note: This is elapsed time for all to compute.
Options per sec.: 1076.329699
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 5.782880E-04
Average reserve: 3.871013
Test passed
Built target run_gpu
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
