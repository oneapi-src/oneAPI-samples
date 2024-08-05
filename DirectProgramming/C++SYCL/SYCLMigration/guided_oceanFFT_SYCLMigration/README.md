# `OceanFFT` Sample

The `OceanFFT` sample simulates an Ocean heightfield using oneMKL FFT functionality. The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to migrate and map SYCL oneMKL FFT equivalent of CUFFT API's
| Time to complete       | 15 minutes
| Category               | Concepts and Functionality

>**Note**: This sample is migrated from NVIDIA CUDA sample. See the [oceanFFT](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/oceanFFT) sample in the NVIDIA/cuda-samples GitHub.

## Purpose

OceanFFT is a simulation of ocean waves/heightfields using FFT. FFT transforms the data from the time/space domain to frequency and vice versa. The ocean is made of many waves all added together. The main principle of Ocean rendering is that it can be modeled as sum of infinite waves at different amplitudes traveling in different directions. 

The cuFFT library allows the floating-point power and parallelism of the GPU without having to develop a custom GPU-based FFT implementation. The equivalent lib on SYCL would be oneAPI Math Kernel Library where FFT function supports one, two, or three dimensions with mixed radices. The supported functions include complex-to-complex and real-to-complex transforms of arbitrary length in single-precision and double-precision. 

> **Note**: The sample used the open-source [SYCLomatic tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html) that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html#gs.5g2aqn) available to augment Base Toolkit.

This sample contains two versions of the code in the following folders:

| Folder Name          | Description
|:---                  |:---
|`01_dpct_output`      | Contains output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that must be manually fixed to get full functionality. (The code does not functionally work as generated.)
|`02_sycl_migrated`    | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br>Intel® Gen11 <br>Intel® Xeon CPU <br>Intel® Data Center GPU Max <br> Nvidia Tesla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software              | SYCLomatic (Tag - 20230720) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.0 <br> oneAPI for NVIDIA GPU plugin from Codeplay (to run SYCL™ applications on NVIDIA® GPUs)

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.<br> How to run SYCL™ applications on NVIDIA® GPUs, refer to 
[oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from Codeplay.


## Key Implementation Details
This sample demonstrates the migration of the following prominent CUDA features:
 - CUDA FFT API (CUFFT)

OceanFFT sample demonstrates how to use FFT API to synthesize and render an ocean surface in real time. SYCL oceanFFT generates wave heightfield at the time based on initial heightfield and dispersion relationship. Then FFT API is executed in inverse order to convert to spatial domain. Finally, height map values and slope values are evaluated and compared with reference values to determine whether it is correct or wrong.

>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.
## CUDA source code evaluation

The OceanFFT sample demonstrates the FFT Computations through different processes one after the another. 

- Generate wave spectrum in frequency domain.
- Execute inverse FFT to convert to spatial domain.
- Update height map values based on output of FFT.
- Calculate slope by partial differences in spatial domain.


> **Note**: For more information on how to use Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.


## Migrate the Code Using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the concurrentKernels sample directory.
   ```
   cd cuda-samples/Samples/4_CUDA_Libraries/oceanFFT/
   ```
3. Generate a compilation database with intercept-build.
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function 
   ```
## Manual Workarounds
The following manual change has been done in order to complete the migration.
   
1. Few CUDA headers are not migrated to SYCL which contain some macros which are used in code. It has been changed manually.
      ```
      CUDART_SQRT_HALF_F
      ```
      Manually defined as below
      ```
      #define SYCLRT_SQRT_HALF_F 0.707106781f
      ```
> **Note**: OceanFFT CUDA sample includes OpenGL feature as well, Since SYCL does not support OpenGL we do not migrate OpenGL  functions.

## Build and Run the `oceanFFT` Sample

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
2. For Nvidia GPUs, install the opensource oneMKL lib and set the environment variables before build.  
   ```
   export ONEMKL_INSTALL_DIR=path_to_opensource_oneMKL_build_dir
   export LD_LIBRARY_PATH=$ONEMKL_INSTALL_DIR/lib:$LD_LIBRARY_PATH
   ```
3. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```
   > **Note**:
   > - By default, no flags are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU.
   > - Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance.
   > - Enable **NVIDIA_GPU** flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs )

 
   By default, this command sequence will build the  `02_sycl_migrated` version of the program.

4. Run `02_sycl_migrated` on GPU.
   ```
   make run_sm
   ```
   Run `02_sycl_migrated` on CPU.
   ```
   export ONEAPI_DEVICE_SELECTOR=opencl:cpu
   make run_sm
   unset ONEAPI_DEVICE_SELECTOR
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
