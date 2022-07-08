# `ISO2DFD` Sample

The ISO2DFD sample refers to Two-Dimensional Finite-Difference Wave Propagation in Isotropic Media.  It is a two-dimensional stencil to simulate a wave propagating in a 2D isotropic medium and illustrates the basics of SYCL*-compliant code using direct programming.

> **Note**: You can find a complete code walk-through of this sample at [Code Sample: Two-Dimensional Finite-Difference Wave Propagation in Isotropic Media (ISO2DFD) – An Intel&reg; oneAPI DPC++ Compiler Example](https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd).

For comprehensive information in using oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide), and use search or the table of contents to find relevant information.

| Property                     | Description
|:---                               |:---
| What you will learn               | How to offload the computation to GPU using Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 10 minutes

## Purpose

ISO2DFD is a finite difference stencil kernel for solving the 2D acoustic isotropic wave equation.  In
this sample, we chose the problem of solving a Partial Differential Equation (PDE), using a
finite-difference method, to illustrate the essential elements of SYCL* queues, buffers, accessors, and kernels. Use it as an entry point to start SYCL* programming or as a
proxy to develop or better understand complicated code for similar problems.

The sample will explicitly run on the GPU as well as CPU to calculate a
result. The output will include GPU device name. The results from the two devices are compared, and if the sample ran correctly report a success message. The output of the wavefield can be plotted using the SU Seismic processing library, which has utilities to display seismic wavefields. You can download the processing library from [John Stockwell’s SeisUnix GitHub](https://github.com/JohnWStockwellJr/SeisUnix).

## Prequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

SYCL implementation explained.

- SYCL queues (including device selectors and exception handlers).
- SYCL buffers and accessors.
- The ability to call a function inside a kernel definition and pass accessor arguments as pointers. A function called inside the kernel performs a computation (it updates a grid point specified by the global ID variable) for a single time step.

##  Build the `iso2dfd` Program for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Include Files

The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Use Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
Perform the following steps:
1. Build the program using the following `cmake` commands.
   ```
    cd iso2dfd_dpcpp &&
    mkdir build &&
    cd build &&
    cmake .. &&
    make -j
    ```
2. Run the program.
    ```
    make run
    ```
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### On Windows* Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019.
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

### Run Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

## Run the Sample
### Application Parameters

You can  execute the code with different parameters. 

Usage: `./iso2dfd n1 n2 Iterations`

Where:
- `n1 n2`: Grid sizes for the stencil
- `Iterations`: Number of timesteps.

For example the following command will run the iso2dfd executable using a 1000x1000 grid size and it will iterate over 2000 time steps.
```
./iso2dfd 1000 1000 2000
```
> **Note**: You can find graphical output for sample execution in [Code Sample: Two-Dimensional Finite-Difference Wave Propagation in Isotropic Media (ISO2DFD) – An Intel&reg; oneAPI DPC++ Compiler Example](https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd).

### Example of Output
```
Initializing ...
Grid Sizes: 1000 1000
Iterations: 2000

Computing wavefield in device ..
Running on Intel(R) Gen9 HD Graphics NEO
The Device Max Work Group Size is : 256
The Device Max EUCount is : 24
SYCL time: 3282 ms

Computing wavefield in CPU ..
Initializing ...
CPU time: 8846 ms

Final wavefields from device and CPU are equivalent: Success
Final wavefields (from device and CPU) written to disk
Finished.
[100%] Built target run
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).