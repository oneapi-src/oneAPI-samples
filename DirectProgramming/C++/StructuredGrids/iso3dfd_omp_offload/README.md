# `ISO3DFD OpenMP Offload` Sample

The `ISO3DFD OpenMP Offload` Sample references three-dimensional finite-difference wave propagation in isotropic media (ISO3DFD). ISO3DFD is a three-dimensional stencil to simulate a wave propagating in a 3D isotropic medium and shows some common challenges and techniques when targeting OpenMP* (OMP) Offload devices, like GPUs, in more complex applications to achieve good performance.

| Area                    | Description
|:---                     |:---
| What you will learn     | How to offload computations to a GPU using OpenMP*
| Time to complete        | 15 minutes


## Purpose

ISO3DFD is a finite difference stencil kernel for solving the 3D acoustic isotropic wave equation. The equation can be used as a proxy for propagating a seismic wave. In this sample, kernels are implemented as 16th order in space, with symmetric coefficients, and 2nd order in time scheme without boundary conditions. Using OpenMP Offload, the sample can explicitly run on the GPU to propagate a seismic wave, which is a compute-intensive task.

The code will attempt to find an available GPU or OpenMP Offload capable device and exit if a compatible device is not detected. By default, the output will print the device name where the OpenMP Offload code ran along with the grid computation metrics - flops and effective throughput. For validating results, an OpenMP/CPU-only version of the application will be run on host/CPU, and results will be compared to the OpenMP Offload version.

The code also demonstrates some of the common optimization techniques that can be used to improve 3D-stencil code running on a GPU device.

## Prerequisites

| Optimized for          | Description
|:---                    |:---
| OS                     | Ubuntu* 18.04
| Hardware               | Skylake with GEN9 or newer
| Software               | Intel® oneAPI DPC++/C++ Compiler


## Key Implementation Details

The basic OpenMP Offload implementation explained in the code includes the following concepts:
- OpenMP offload target data map construct.
- **Default Baseline version** demonstrates the use of OpenMP offload target parallel for construct with the collapse.
- **Optimized version 1** demonstrates the use of OpenMP offload teams distribute construct and use of `num_teams` and `thread_limit` clause.
- Incremental **Optimized version 2** demonstrates the use of OpenMP offload teams distribute construct with improved data-access pattern.
- Incremental **Optimized version 3** demonstrates use of OpenMP CPU threads along with OpenMP offload target construct.

  **Performance number tabulation**

  | ISO3DFD Version                | Performance Data
  |:---                            |:---
  | Default Baseline version       | 1.0
  | Optimized version 1	           | 1.11x
  | Optimized version 2	           | 1.48x
  | Optimized version 3	           | 1.60x

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `ISO3DFD OpenMP Offload` Program

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
    d build
    make ..
    make
    ```

    By default, the executable is built with the default baseline version. Optionally, you can build the kernel with optimized versions.
    ```
    cmake -DUSE_OPT1=1 ..
    make -j
    ```
    ```
    cmake -DUSE_OPT2=1 ..
    make -j
    ```
    ```
    cmake -DUSE_OPT3=1 ..
    make -j
    ```
    If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
    ```
    make VERBOSE=1
    ```
3. Change the optimized version (if needed)

   If you already compiled with the optimization path, for example "*cmake -DUSE_OPT1=1 ..*", use "*cmake -DUSE_OPT1=0 ..*" can go back to the baseline version.

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `ISO3DFD OpenMP Offload` Program

### Configurable Parameters

The program supports several configurable input parameters. The general syntax is as follows:
```
src/iso3dfd n1 n2 n3 n1_block n2_block n3_block Iterations
```

|Parameter                      |Description
|:---                           |:---
|n1 n2 n3                       |Grid sizes for the stencil. The sample uses `256 256 256` as the default values.
|n1_block n2_block n3_block     |Cache block sizes for **CPU** or **tile sizes** for OpenMP Offload. The sample uses as `16 8 64` the default values.
|Iterations                    	|Number of timesteps. The sample uses `100` as the default value.

The default syntax is `src/iso3dfd 256 256 256 16 8 64 100`.

### On Linux

1. Run the program.
   ```
   make run
   ```
2. Clean the program. (Optional)
   ```
   make clean
   ```


### On Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

|Available Nodes	    |Command Options
|:---                 |:---
|GPU	                |`qsub -l nodes=1:gpu:ppn=2 -d .`
|CPU	                |`qsub -l nodes=1:xeon:ppn=2 -d .`

For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

For more information on using Intel® DevCloud, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)

You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`. 

If you choose to use scripts, jobs terminate with writing files to the disk:
- `<script_name>.sh.eXXXX`, which is the job stderr
- `<script_name>.sh.oXXXX`, which is the job stdout

Here XXXX is the job ID, which gets printed to the screen after each qsub command.

You can inspect output of the sample.
```
cat run.sh.oXXXX
```
Once the jobs complete, you can remove the stderr and stdout files.
```
rm run.sh.*
```
#### Build and Run on Intel® DevCloud

1. Open a terminal on a Linux* system.
2. Log in to the Intel® DevCloud.
   ```
   ssh devcloud
   ```
3. Download the samples from GitHub.
   ```
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```
4. Change to the sample directory.
5. Configure the sample for the appropriate node.

   The following example is for a GPU node. (This is a single line script.)
	```
	qsub  -I  -l nodes=1:gpu:ppn=2 -d .
	```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
   - `-d .` makes the current folder as the working directory for the task.

6. Perform build steps you would on Linux. (Including optionally cleaning the project.)
7. Run the sample.

> **Note**: To inspect job progress if you are using a script, use the qstat utility.
>   ```
>   watch -n 1 qstat -n -1
>   ```
>  The command displays the results every second. The job is complete when no new results display.

8. Review the output.
9. Disconnect from Intel® DevCloud.
	```
	exit
	```

## Example Output

### Example Output for Baseline Version

```
Scanning dependencies of target run
Grid Sizes: 256 256 256
Tile sizes ignored for OMP Offload
--Using Baseline version with omp target with collapse
Memory Usage (MBytes): 230
--------------------------------------
time         : 4.118 secs
throughput   : 407.412 Mpts/s
flops        : 24.8521 GFlops
bytes        : 4.88894 GBytes/s

--------------------------------------

--------------------------------------
Checking Results ...
Final wavefields from OMP Offload device and CPU are equivalent: Success
--------------------------------------
```

### Example Output for Optimized Version 3
```
Grid Sizes: 256 256 256
Tile sizes: 16 8 64
Using Optimized target code - version 3:
--OMP Threads + OMP_Offload with Tiling and Z Window
Memory Usage (MBytes): 230
--------------------------------------
time         : 3.014 secs
throughput   : 556.643 Mpts/s
flops        : 33.9552 GFlops
bytes        : 6.67971 GBytes/s

--------------------------------------

--------------------------------------
Checking Results ...
Final wavefields from OMP Offload device and CPU are equivalent: Success
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).