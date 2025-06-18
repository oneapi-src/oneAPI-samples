# `MPI Communications Using GPU Buffers` Sample

The `MPI Communications Using GPU Buffers` sample demonstrates how to use GPU support functionality available in Intel® MPI Library.

| Area                 | Description
|:---                  |:---
| What you will learn  | How to use Intel® MPI Library with GPU buffers
| Time to complete     | 15 minutes
| Category             | Concepts and Functionality

For more information on Intel® MPI Library and complete documentation of all features,
see the [Intel® MPI Library Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-documentation.html) page.

## Purpose

The sample demonstrates the basic communication use case for GPU buffers and the overall concept of the GPU-aware Intel® MPI Library.

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux*
| Hardware            | 4th Generation Intel® Xeon® Scalable Processors <br> Intel® Data Center GPU Max Series
| Software            | Intel® MPI Library


## Key Implementation Details

This sample uses two approaches (OpenMP and SYCL*) in separate source files offloading code to a GPU. The sample also uses Intel® MPI [GPU Support feature](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/current/gpu-support.html). The sample uses the OpenMP [target](https://www.openmp.org/spec-html/5.0/openmpsu60.html) construct and SYCL [parallel_for](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/data-parallelism-in-c-using-sycl.html) function.

To pass a pointer of an offloaded memory region to MPI, you may need to use specific compiler directives or get it from the corresponding acceleration runtime API.

- `use_device_ptr` and `use_device_addr` are useful keywords to obtain device pointers in the OpenMP environment, as demonstrated in the given example.

- Memory allocated with SYCL memory allocation functions (for example, `sycl::malloc_device`) may be directly passed to MPI communication functions.

## Build the `MPI Communications Using GPU Buffers` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system-wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, or if you are using a Unified Directory Layout, see
*[Use the setvars and oneapi-vars Scripts with Linux*](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/use-the-setvars-script-with-linux-or-macos.html)*.

### On Linux*

1. Change to the sample directory.

2. Run `make` to build a release version of the sample.
   ```
   make
   ```
   Alternatively, you can build the debug version.
   ```
   make debug
   ```

3. Clean the project files. (Optional)
   ```
   make clean
   ```

### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the Diagnostics Utility. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html)* for more information on using the utility.

## Run the `MPI Communications Using GPU Buffers` Sample

### On Linux

1. Run the OpenMP version of the sample using the following `mpirun` command:
   ```
   mpirun -n 2 -genv I_MPI_OFFLOAD=1 -genv LIBOMPTARGET_PLUGIN=level0 ./src/mpi_send_gpu_omp
   ```

2. Run the SYCL version of the sample using the following `mpirun` command:
   ```
   mpirun -n 2 -genv I_MPI_OFFLOAD=1 -genv ONEAPI_DEVICE_SELECTOR=level_zero:* ./src/mpi_send_gpu_sycl
   ```

If everything worked, the ACTIVE_RANK (by default defined as 1) will generate sample data and transfer it to the peer rank. The peer rank should verify the data and report any errors.

## Example Output

```
mpiexec -n 2 -genv I_MPI_OFFLOAD=1 -genv LIBOMPTARGET_PLUGIN=level0 ./src/mpi_send_gpu_omp
[0] Receiving data from rank 1
[1] Sending GPU buffer 0xff00fffffffc0180 to rank 0
[0] result: 0 2 2 4 6 10 16 26 42 68
[0] SUCCESS
```

## License

Code samples are licensed under the MIT license. See [License.txt](License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
