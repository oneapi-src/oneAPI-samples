# `DPC Reduce` Sample

The `DPC Reduce` is a simple program that calculates pi (&pi;) using several methods. This code sample demonstrates how to incorporate SYCL* standards into an MPI program.

| Property                | Description
|:---                     |:---
| What you will learn     | How to perform reduction on CPU and GPU using the Intel® oneAPI DPC++ Library (oneDPL)
| Time to complete        | 30 minutes

## Purpose
It is a well known mathematical formula that if you integrate from 0 to 1 over the function (4.0 / (1+x*x))dx the answer is pi (&pi;). One can approximate this integral by summing up the area of a large number of rectangles over this same range.

The different methods shown in this sample use a simple calculation for pi. Each function calculates pi by breaking the range into many tiny rectangles and then summing up the results.

This example demonstrates how to perform reduction through several methods using:
- a CPU in serial mode
- a CPU in parallel mode using Intel® oneAPI Threading Building Blocks (oneTBB)
- a GPU using direct SYCL* compliant C++ coding
- a GPU using multiple steps with Intel® oneAPI DPC++ Library (oneDPL) algorithms to transform and reduce
- the oneDPL transform_reduce algorithm

The parallel computations use oneTBB and oneDPL. This sample demonstrates how to incorporate SYCL* standards into an MPI program. The code sample runs multiple MPI ranks to distribute the calculation of the number Pi. Each rank offloads the computation to an accelerator (GPU/CPU) to compute a partial computation of the number Pi.

## Prerequisites
| Optimized for         | Description
|:---                   |:---
| OS	                  | Ubuntu* 18.04
| Hardware	            | Skylake with GEN9 or newer
| Software	            | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic SYCL* implementation explained in the code includes accessor,
kernels, queues, buffers. The sample demonstrated using some oneDPL calls.

> **Note**: For comprehensive information about oneAPI programming, see the [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Building the `DPC Reduce` Program for CPU and GPU
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Use Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)

### On Linux*
1. Change to the sample directory.
2. Specify the correct compiler for the Intel® MPI Library.
   ```
   export I_MPI_CXX=icpx
   ```
2. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Run the `DPC Reduce` Sample
### Application Parameters
This sample allows you to input a specific MPI rank.

Usage: `mpirun -n <num> ./dpc_reduce`

where

- `<num>`: Indicates the MPI rank number. Excluding  -n <num> will show all ranks and devices.

### On Linux
1. Run the program.
   ```
   mpirun ./dpc_reduce
   ```
   Alternatively, you can run the program using the following command: `make run`.

> **Note**: If you run the sample on a CPU as your default device, you might need to increase
the memory allocation for OpenCL™. If that is the case, then adjust the value by setting an environment variable of `CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=16MB ./dpc_reduce`.

2. Clean the program. (Optional)
   ```
   make clean
   ```

## Example Output
```
Rank #0 runs on: testsystem, uses device: Intel(R) Gen9 HD Graphics NEO
Number of steps is 1000000
Cpu Seq calc:               PI =3.14 in 0.00422 seconds
Cpu TBB  calc:              PI =3.14 in 0.00177 seconds
oneDPL native:              PI =3.14 in 0.209 seconds
oneDPL native2:             PI =3.14 in 0.213 seconds
oneDPL native3:             PI =3.14 in 0.00222 seconds
oneDPL native4:             PI =3.14 in 0.00237 seconds
oneDPL two steps:           PI =3.14 in 0.0014 seconds
oneDPL transform_reduce:    PI =3.14 in 0.000528 seconds
mpi native:                 PI =3.14 in 0.548 seconds
mpi transform_reduce:       PI =3.14 in 0.000498 seconds
success
```

## License
Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
