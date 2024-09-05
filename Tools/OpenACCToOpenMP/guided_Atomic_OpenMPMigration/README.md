# `Atomic` Sample
 
This sample illustrates the read, write, update & capture clauses for the atomic directive. The original OpenACC source code is migrated to OpenMP to Offload on Intel® Platforms.

| Area                  | Description
|:---                       |:---
| What you will learn       | Migrating and optimizing Atomic from OpenACC to OpenMP
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

## Purpose

OpenMP atomic operations allows multiple threads to safely update a shared numeric variable, such as on hardware platforms that support atomic operation use. An atomic operation applies only to the single assignment statement that immediately follows it, so atomic operations are useful for code that requires fine-grain synchronization.

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
- #pragma acc parallel loop copy() copyout()

  The kernels construct identifies a region of code that may contain parallelism that has been as been translated into:
  - #pragma omp target teams loop map(tofrom:) map(from:)
- #pragma acc atomic read

  The `atomic read` reads the value of a variable atomically. The value of a shared variable can be read safely, avoiding the danger of reading an intermediate value of the variable when it is accessed simultaneously by a concurrent thread. This has been translated into:
  - #pragma omp atomic read
- #pragma acc atomic write

  The `atomic write` writes the value of a variable atomically. The value of a shared variable can be written exclusively to avoid errors from simultaneous writes. This has been translated into:
  - #pragma omp atomic write
- #pragma acc atomic update

  The `atomic update` updates the value of a variable atomically. Allows only one thread to write to a shared variable at a time, avoiding errors from simultaneous writes to the same variable. This has been translated into:
  - #pragma omp atomic update
- #pragma acc atomic capture

  The `atomic capture` updates the value of a variable while capturing the original or final value of the variable atomically. This has been translated into:
  - #pragma omp atomic capture
  

>  **Note**: Refer to [Portability across Heterogeneous Architectures](https://www.intel.com/content/www/us/en/developer/articles/technical/openmp-accelerator-offload.html#gs.n33nuz) for general information about the migration of OpenACC to OpenMP.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that the compiler, libraries, and tools are ready for development.

## Migrate the `Atomic` Sample

### Migrate the Code using intel-application-migration-tool-for-openacc-to-openmp

For this sample, the tool takes application sources (either C/C++ or Fortran languages) with OpenACC constructs and generates a semantically-equivalent source using OpenMP. Follow these steps to migrate the code

  1. Tool installation
     ```
     git clone https://github.com/intel/intel-application-migration-tool-for-openacc-to-openmp.git
     ```

The binary of the translator can be found inside intel-application-migration-tool-for-openacc-to-openmp/src location
    
  2. The openacc sample is taken from [Openacc-samples](https://github.com/OpenACC/openacc-examples.git)
     ```
     cd openacc-examples/Submissions/C/Atomic/
     ```
  3. Now invoke the translator to migrate the openACC pragmas to OpenMP as shown below
     ```
     intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp atomic.c
     ```
For each given input-file, the tool will generate a translation file named <input-file>.translated and will also dump a report with translation details into a file named <input-file>.report.

## Build the `Atomic` Sample for GPU

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
