# `BilateralFilter` Sample
 
This sample uses OpenMP directives to perform a simple bilateral filter on an image and measures performance. The original OpenACC source code is migrated to OpenMP to Offload on Intel® Platforms.

| Area                  | Description
|:---                       |:---
| What you will learn       | Migrating and optimizing bilateralFilter from OpenACC to OpenMP
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

## Purpose

Bilateral filter is an edge-preserving nonlinear smoothing filter. There are three parameters distribute to the filter: gaussian delta, euclidean delta and iterations.
When the euclidean delta increases, most of the fine texture will be filtered away, yet all contours are as crisp as in the original image. If the euclidean delta approximates to ∞, the filter becomes a normal gaussian filter. Fine texture will blur more with larger gaussian delta. Multiple iterations have the effect of flattening the colors in an image considerably, but without blurring edges, which produces a cartoon effect.

> **Note**: We use intel-application-migration-tool-for-openacc-to-openmp which assists developers in porting OpenACC code automatically to OpenMP code. 

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:--- 
| `openMP_migrated_output`            | Contains the OpenMP migrated code.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 22.04
| Hardware                   | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Data Center GPU Max
| Software                   | Intel oneAPI Base Toolkit version 2024.2 <br> intel-application-migration-tool-for-openacc-to-openmp

For more information on how to install the above Tool, visit [intel-application-migration-tool-for-openacc-to-openmp](https://github.com/intel/intel-application-migration-tool-for-openacc-to-openmp)

## Key Implementation Details

This sample demonstrates the migration of the following OpenACC pragmas: 
- #pragma acc kernels copyin() copyout() create() if()
  The kernels construct identifies a region of code that may contain parallelism that has been as been translated into:
  - #pragma omp target map(to:) map(from:) map(alloc:) if()
- #pragma acc loop independent, gang
  The loop directive is intended to give the compiler additional information about the next loop in the code. This has been translated into:
  - #pragma omp loop order(concurrent)
  

>  **Note**: Refer to [Portability across Heterogeneous Architectures](https://www.intel.com/content/www/us/en/developer/articles/technical/openmp-accelerator-offload.html#gs.n33nuz) for general information about the migration of OpenACC to OpenMP.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that the compiler, libraries, and tools are ready for development.

## Migrate the `bilateralFilter` Sample

### Migrate the Code using intel-application-migration-tool-for-openacc-to-openmp

For this sample, the tool takes application sources (either C/C++ or Fortran languages) with OpenACC constructs and generates a semantically-equivalent source using OpenMP. Follow these steps to migrate the code

  1. Tool installation
     ```
     git clone https://github.com/intel/intel-application-migration-tool-for-openacc-to-openmp.git
     ```

The binary of the translator can be found inside intel-application-migration-tool-for-openacc-to-openmp/src location
    
  2. The openacc sample is taken from NVIDIA_HPC_SDK samples and can be found at the installation location as shown below
     ```
     cd /opt/hpc_software/sdk/nvidia/hpc_sdk/Linux_x86_64/24.3/examples/OpenACC/SDK/src/bilateralFilter
     ```
  3. Now invoke the translator to migrate the openACC pragmas to OpenMP as shown below
     ```
     intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp bilateralFilter.c
     ```
For each given input-file, the tool will generate a translation file named <input-file>.translated and will also dump a report with translation details into a file named <input-file>.report.

### Optimization

The tool does not aim at guaranteeing the best achievable performance but at generating a semantically equivalent translation. To optimize the code, one can use `teams` directive as it plays a crucial role, especially in the context of offloading computations to devices such as GPUs. The `teams` directive in OpenMP is used to create a league of thread teams, each of which can execute concurrently that leads to good performance.
Introduce teams in the pragma as shown below in the translated code (line 126)
   ```
   #pragma omp target teams map(to:h_Src[0:imageW*imageH],h_Gaussian[0:KERNEL_LENGTH])\
            map(from:h_Dst[0:imageW*imageH]) map(alloc:h_BufferX[0:imageW*imageH],\
            h_BufferY[0:imageW*imageH],h_BufferZ[0:imageW*imageH],\
            h_BufferW[0:imageW*imageH]) if(accelerate)
   ```

## Build the `bilateralFilter` Sample for GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ make
   ```
   
By default, this command sequence will build the `openMP_migrated_output ` version of the program.

3. Run the program.
   ```
   $ make run
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
