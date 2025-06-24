# `MonteCarloMultiGPU` Sample

This sample uses OpenMP directives to implement a Monte Carlo simulation to estimate the price of a European call option and its confidence interval. The original OpenACC source code is migrated to OpenMP to Offload on Intel® Platforms.

| Area                  | Description
|:---                       |:---
| What you will learn       | Migrating and optimizing MonteCarloMultiGPU from OpenACC to OpenMP
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

## Purpose

The purpose of this sample is to estimate the expected value of a European call option and provide a measure of the confidence interval around this estimate using Monte Carlo simulation. Monte Carlo simulation is a statistical technique that uses random sampling to approximate complex mathematical and physical systems. This particualr sample demonstartes how it can be offloaded to GPUs using openmp directives and obtain good performance.

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

This sample demonstrates the migration of the following OpenACC pragmas & APIs:
- #pragma acc routine
- #pragma acc enter data create() copyin()
- #pragma acc kernels loop async() present()
- #pragma acc loop reduction()
- #pragma acc exit data copyout() delete()
- acc_get_num_devices()
- acc_wait()
- acc_wait_all()
- acc_get_num_devices()
- acc_set_device_num()


>  **Note**: Refer to [Portability across Heterogeneous Architectures](https://www.intel.com/content/www/us/en/developer/articles/technical/openmp-accelerator-offload.html#gs.n33nuz) for general information about the migration of OpenACC to OpenMP.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that the compiler, libraries, and tools are ready for development.

## Migrate the `MonteCarloMultiGPU` Sample

### Migrate the Code using intel-application-migration-tool-for-openacc-to-openmp

For this sample, the tool takes application sources (either C/C++ or Fortran languages) with OpenACC constructs and generates a semantically-equivalent source using OpenMP. Follow these steps to migrate the code

  1. Tool installation
     ```
     git clone https://github.com/intel/intel-application-migration-tool-for-openacc-to-openmp.git
     ```

The binary of the translator can be found inside intel-application-migration-tool-for-openacc-to-openmp/src location

  2. The openacc sample is taken from NVIDIA_HPC_SDK samples and can be found at the installation location as shown below
     ```
     cd /opt/hpc_software/sdk/nvidia/hpc_sdk/Linux_x86_64/24.3/examples/OpenACC/SDK/src/MonteCarloMultiGPU
     ```
  3. Now invoke the translator to migrate the openACC pragmas to OpenMP as shown below
     ```
     intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp MonteCarloMultiGPU.c
     ```
For each given input-file, the tool will generate a translation file named <input-file>.translated and will also dump a report with translation details into a file named <input-file>.report.

### Manual workarounds

The tool recognizes many of the OpenACC compute and data constructs but cannot translate OpenACC API calls at the moment. One can check the generated report file which suggests the corresponsding OpenMP equivalents of the unmigrated OpenACC API calls as shown below

```
Note the following OpenACC API calls. These have NOT been translated.
* Line 125 contains an invocation to 'acc_get_num_devices'. You can use omp_get_num_devices().
* Line 172 contains an invocation to 'acc_wait'. Consider using #pragma omp taskwait.
* Line 172 contains an invocation to 'acc_wait_all'. Consider using #pragma omp taskwait.
* Line 185 contains an invocation to 'acc_get_num_devices'. You can use omp_get_num_devices().
* Line 208 contains an invocation to 'acc_set_device_num'. You can use omp_set_default_device().
```

## Build the `MonteCarloMultiGPU` Sample for GPU

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
[License.txt](License.txt) for details.

Third party program licenses are at [third-party-programs.txt](third-party-programs.txt).
