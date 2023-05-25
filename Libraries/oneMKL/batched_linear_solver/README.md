# Batched Linear Solver Sample
Solving a batch of linear systems is a common operation in engineering and scientific computing.
Consequently, the oneMKL LAPACK implementation provides batched LU solver functions that are optimized
for Intel processors.

For more information on oneMKL, and complete documentation of all oneMKL routines, see https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04
| Hardware            | Intel&reg; Skylake with Gen9 or newer
| Software            | Intel&reg; oneMKL, Intel&reg; Fortran Compiler
| What you will learn | How to optimize host-device data transfer when using OpenMP target offload
| Time to complete    | 45 minutes

## Purpose
This sample shows how to solve a batch of linear systems using the batched solver functions
([``getrf_batch_strided``](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2023-1/getrf-batch-strided.html) and [``getrs_batch_strided``](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2023-1/getrs-batch-strided.html)), how to offload these functions to an accelerator using the OpenMP target directives, and how to minimize host-device data transfer to achieve better performance. The following article provides more detailed descriptions of the sample codes: [Solving Linear Systems Using oneMKL and OpenMP Target Offloading](https://www.intel.com/content/www/us/en/developer/articles/technical/solve-linear-systems-onemkl-openmp-target-offload.html).

## Key Implementation Details
In general, the factored matrices and pivots can be discarded after the linear systems are solved. Only the solutions
need to be transferred back to the host. Two sample codes are provided: `lu_solve_omp_offload.F90` and
`lu_solve_omp_offload_optimized.F90`. The first shows a straightforward way to
dispatch the LU factorization and solver functions using two OpenMP target regions. The code gives correct results
but performs some unnecessary host-device data transfer. The second sample code fuses the two OpenMP regions to
improve performance by minimizing data transfer.

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

## Building and Running the Batched Linear Solver Sample
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
>For more information on environment variables, see [Use the setvars Script for Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Running Samples on the DevCloud
When running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

Run `make` to build and run the sample. Three programs are generated: 

1. `lu_solve`: CPU-only, OpenMP disabled
2. `lu_solve_omp_offload`: two OpenMP target regions with accelerator offload enabled
3. `lu_solve_omp_offload_optimized`: one OpenMP target offload region to minimize host-device data transfer

Note that the makefile only runs small tests to verify that the executables are working correctly. The problem sizes are too small to justify accelerator offload. Use the following command-line options to run the tests shown in [Solving Linear Systems Using oneMKL and OpenMP Target Offloading](https://www.intel.com/content/www/us/en/developer/articles/technical/solve-linear-systems-onemkl-openmp-target-offload.html): `-n 16000 -b 8 -r 1 -c 5`.

You can remove all generated files with `make clean`.

### Example of Output
If everything is working correctly, the output should be similar to this:
```
u172874@s001-n157:~/oneAPI-samples/Libraries/oneMKL/batched_linear_solver$ make
ifx lu_solve_omp_offload.F90 -o lu_solve -i8 -free -qmkl
ifx lu_solve_omp_offload.F90 -o lu_solve_omp_offload -i8 -free -qmkl -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -fsycl -L/glob/development-tools/versions/oneapi/2023.0.1/oneapi/mkl/2023.0.0/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl
ifx lu_solve_omp_offload_optimized.F90 -o lu_solve_omp_offload_optimized -i8 -free -qmkl -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -fsycl -L/glob/development-tools/versions/oneapi/2023.0.1/oneapi/mkl/2023.0.0/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl
./lu_solve -n 64 -b 8 -r 1 -c 2
 Matrix dimensions:                    64
 Batch size:                     8
 Number of RHS:                     1
 Number of test cycles:                     2
 Computation completed successfully  2.849000000000000E-002 seconds
 Computation completed successfully  4.600000000000000E-005 seconds
 Total time:  2.853600000000000E-002 seconds
./lu_solve_omp_offload -n 64 -b 8 -r 1 -c 2
 Matrix dimensions:                    64
 Batch size:                     8
 Number of RHS:                     1
 Number of test cycles:                     2
 Computation completed successfully   1.52985400000000      seconds
 Computation completed successfully  1.212700000000000E-002 seconds
 Total time:   1.54198100000000      seconds
./lu_solve_omp_offload_optimized -n 64 -b 8 -r 1 -c 2
 Matrix dimensions:                    64
 Batch size:                     8
 Number of RHS:                     1
 Number of test cycles:                     2
 Computation completed successfully   1.54803900000000      seconds
 Computation completed successfully  1.202200000000000E-002 seconds
 Total time:   1.56006100000000      seconds
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2023-1/overview.html)
