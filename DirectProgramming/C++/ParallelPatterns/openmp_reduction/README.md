# `OpenMP* Reduction` Sample

The `OpenMP* Reduction` code sample is a simple program to calculate pi; however, the program is implemented using C++ and OpenMP* for Intel® CPU and accelerators.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to use OpenMP for CPU and GPU offload
| Time to complete      | 10 min

## Purpose

This example demonstrates how to perform reduction by using the CPU in serial mode, the CPU in parallel mode (using OpenMP), and the GPU using OpenMP offloading.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS	                 | Ubuntu* 18.04
| Hardware	           | Skylake with GEN9 or newer
| Software	           | Intel® oneAPI DPC++ Compiler

## Key Implementation Details

This code shows how to use OpenMP on the CPU host as well as using target offload capabilities.

The different modes use a simple calculation using the well known mathematical formula that states if one integrates from 0 to 1 over the function, $(4.0/(1+x*x))dx$, the answer is pi. One can approximate this integral by summing up the area of a large number of rectangles over this same range.

Each of the different functions calculates pi by breaking the range into many tiny rectangles and then summing up the results.

>**Note**: For comprehensive information about oneAPI programming, see the [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `OpenMP* Reduction` Sample

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
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
1. Change to the sample directory.
2. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
``make VERBOSE=1``

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `OpenMP* Reduction` Program

### On Linux

1. Run the program.
   ```
   make run
   ```
2. Clean the program. (Optional)
   ```
   make clean
   ```

### Build and Run the `OpenMP* Reduction` Sample in Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. You can specify a node using a single line script.

```
qsub  -I  -l nodes=1:xeon:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:xeon:ppn=2` (lower case L) assigns a CPU node.
- `-d .` makes the current folder as the working directory for the task.

|Available Nodes	    |Command Options
|:---                   |:---
|GPU	                | `qsub -l nodes=1:gpu:ppn=2 -d .`
|CPU	                | `qsub -l nodes=1:xeon:ppn=2 -d .`

For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

For more information on using Intel® DevCloud, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).


## Example Output

The actual results depend on the target device.

```
Number of steps is 1000000
Cpu Seq calc:           PI =3.14 in 0.00105 seconds
Host OpenMP:            PI =3.14 in 0.0010 seconds
Offload OpenMP:         PI =3.14 in 0.0005 seconds
success
Built target run
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).