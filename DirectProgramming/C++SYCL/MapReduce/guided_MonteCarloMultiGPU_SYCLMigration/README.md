# `MonteCarloMultiGPU` Sample

The `MonteCarloMultiGPU` sample evaluates fair call price for a given set of European options using the Monte Carlo approach. MonteCarlo simulation is one of the most important algorithms in quantitative finance. This sample uses a single CPU Thread to control multiple GPUs.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to migrate and map SYCL RNG philox4x32x10<1> equivalent of cuRAND API's
| Time to complete       | 15 minutes
| Category               | Code Optimization

>**Note**: This sample is migrated from NVIDIA CUDA sample. See the [MonteCarloMultiGPU](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/MonteCarloMultiGPU) sample in the NVIDIA/cuda-samples GitHub.


## Purpose
The MonteCarlo Method provides a way to compute expected values by generating random scenarios and then averaging them. The execution of these random scenarios can be parallelized very efficiently. 

With the help of a GPU, we reduce and speed up the problem by parallelizing each path. That we can assign each path to a single thread, simulating thousands of them in parallel, with massive savings in computational power and time.

> **Note**: The sample used the open-source [SYCLomatic tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html) that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html#gs.5g2aqn) available to augment Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                          | Description
|:---                                  |:---
| 01_dpct_output                       | Contains output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that must be manually fixed to get full functionality. (The code does not functionally work as generated.)
| 02_sycl_migrated                     | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br>Intel® Gen11 <br>Intel® Xeon CPU <br>Intel® Data Center GPU Max <br> Nvidia Testa P100 <br> Nvidia A100 <br> Nvidia H100 
| Software              | SYCLomatic (Tag - 20230720) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.0 <br> oneAPI for NVIDIA GPU plugin from Codeplay (to run SYCL™ applications on NVIDIA® GPUs)

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.<br> How to run SYCL™ applications on NVIDIA® GPUs, refer to 
[oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from Codeplay.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features: 
- Random Number Generator

Calls to cuRAND function APIs are being translated to equivalent Intel® oneAPI Math Kernel Library (oneMKL) function calls. 

The `MonteCarloOneBlockPerOption()` kernel is the main computation kernel. It calculates the integral over all paths using a single thread block per option. 

The `rngSetupStates()` kernel initializes the random number generator states for each thread. 

The `initMonteCarloGPU()` function allocates memory on the GPU, sets up the random number generator states, and initializes other variables. 

The `closeMonteCarloGPU()` function computes statistics and deallocates the memory on the GPU. 

Finally, the `MonteCarloGPU()` function performs the main computations by copying data to the GPU, launching the computation kernel, and copying the results back to the host.

>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.

## CUDA source code evaluation

Pricing of European Options can be done by applying the Black-Scholes formula and with MonteCarlo approach.

MonteCarlo Method first generates a random number based on a probability distribution. The random number then uses the additional inputs of volatility and time to expiration to generate a stock price. The generated stock price at the time of expiration is then used to calculate the value of the option. The model then calculates results over and over, each time using a different set of random values from the probability functions

The first stage of the computation is the generation of a normally distributed N(0, 1)number sequence, which comes down to uniformly distributed sequence generation. Once we’ve generated the desired number of samples, we use them to compute an expected value and confidence width for the underlying option. 

The Black-Scholes model relies on fixed inputs (current stock price, strike price, time until expiration, volatility, risk free rates, and dividend yield). The model is based on geometric Brownian motion with constant drift and volatility. We can calculate the price of the European put and call options explicitly using the Black–Scholes formula.

The price of a call option C in terms of the Black–Scholes parameters is

C=N(d1)×S−N(d2)×PV(K)

where:

•	d1=1σ√T[log(SK)+(r+σ22)T]

•	d2=d1−σ√T

•	PV(K)=Kexp(−rT)

After repeatedly computing appropriate averages, the estimated price of options can be obtained, which is consistent with the analytical results from Black-Scholes model.

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the Code Using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the MonteCarloMultiGPU sample directory.
   ```
   cd cuda-samples/Samples/5_Domain_Specific/MonteCarloMultiGPU/
   ```
3. Generate a compilation database with intercept-build.
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The `--in-root` specifies path to the root of the source tree to be migrated.

   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function
   ```

## Build and Run the `MonteCarloMultiGPU` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```
   > **Note**:
   > - By default, no flag are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU.
   > - Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
   > - Enable **NVIDIA_GPU** flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs )
    
   By default, this command sequence will build the `02_sycl_migrated` version of the program.

4. Run `02_sycl_migrated` for CPU and GPU.
     ```
    make run_sm_cpu (runs on cpu)
    make run_sm_gpu (runs on Level-Zero Backend)
    make run_sm_gpu_opencl (runs on OpenCL Backend)
    make run_sm_gpu_cuda (runs on cuda Backend)
    ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-0/overview.html) for more information on using the utility.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
