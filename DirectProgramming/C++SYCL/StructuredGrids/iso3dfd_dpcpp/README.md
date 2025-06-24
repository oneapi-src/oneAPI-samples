# `ISO3DFD` Sample
ISO3DFD refers to three-dimensional finite-difference wave propagation in isotropic media. The `ISO3DFD` sample is a three-dimensional stencil to simulate a wave propagating in a 3D isotropic medium. The sample code shows common challenges when targeting SYCL* devices (GPU/CPU) in complex applications.

| Area                       | Description
|:---                        |:---
| What you will learn        | How to offload complex computations to a GPU
| Time to complete           | 15 minutes

## Purpose
The `ISO3DFD` sample is a finite difference stencil kernel for solving the 3D acoustic isotropic wave equation. The solution can be used as a proxy for propagating a seismic wave.

In this sample, kernels are implemented as 16th order in space, with symmetric coefficients, and 2nd order in time scheme without boundary conditions. The sample code can explicitly run on a GPU or CPU, or on both, to propagate a seismic wave. The calculations are compute intensive.

> **Note**: You can find information about this sample at [Explore SYCL* with Samples from Intel](https://www.intel.com/content/www/us/en/develop/documentation/explore-dpcpp-samples-from-intel/top.html#top_STEP4_ISO3DFD).

## Prerequisites
| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                | Skylake with GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic SYCL* implementation explained in the code includes:
- SYCL local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each workgroup)
- Code for Shared Local Memory (SLM) optimizations
- SYCL kernels (including parallel_for function and nd-range<3> objects)
- SYCL queues (including custom device selector and exception handlers)

The code attempts to run on a compatible GPU first. If a compatible GPU is not found, then the program runs on the CPU.

For validating results, a serial version of the application will be run on CPU, and results will be compared to the SYCL version. In some cases, the code will use a field programmable array (FPGA) emulator to run the SYCL variant.

The program output includes the device name that ran the code along with the grid computation metrics, flops, and effective throughput.

| `ISO3DFD` Sample        | Performance data
|:---                     |:---
| Scalar baseline -O2     | 1.0
| SYCL                    | 2x speedup

## Build the `ISO3DFD` Program for CPU and GPU

### Setting Environment Variables
When working with the Command Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> Microsoft Visual Studio:
> - Open a command prompt window and execute `setx SETVARS_CONFIG " "`. This only needs to be set once and will automatically execute the `setvars` script every time Visual Studio is launched.
>
>For more information on environment variables, see "Use the setvars Script" for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

If you wish to fine tune the list of components and the version of those components, use
a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

### Use Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
 2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
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
   By default, executable builds for a kernel with direct global memory usage. You can build the kernel with shared local memory (SLM) buffers.
   ```
   cmake -DSHARED_KERNEL=1 ..
   make
   ```
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
### On Windows*
**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. Right-click on the project in **Solution Explorer** and select **Rebuild**.
4. From the top menu, select **Debug** > **Start without Debugging**. (This runs the program.)

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild iso3dfd.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `ISO3DFD` Program
### Configurable Application Parameters
You can specify input parameters for the program. Different devices and variants require different inputs.

Usage: `iso3dfd.exe n1 n2 n3 b1 b2 b3 iterations [omp|sycl] [gpu|cpu]`

|Parameter      | Description
|:---           |:---
|`n1 n2 n3`     | Grid sizes for the stencil.
|`b1 b2 b3`     | (OMP) Cache block sizes for cpu openmp version.
|`b1 b2`        | (SYCL) Thread block sizes in X and Y dimensions for SYCL version.
|`b3`           | (SYCL) Size of slice of work in Z dimension for SYCL version.
|`iterations`   | Number of timesteps.
|`omp\|sycl`    | (Optional) Run the OpenMP or the SYCL variant. Default to both for validation.
|`gpu\|cpu`     | (Optional) Device for the SYCL version; default to GPU if available. If a GPU is not available, the program runs on the CPU.

### On Linux
1. Run the program.
   ```
   make run
   ```
   Alternatively, you can select CPU as a SYCL device by using `make run_cpu`.

### On Windows
1. Change to the output directory.
2. Specify the input values, and run the program.
   ```
   iso3dfd_sycl.exe 256 256 256 32 8 64 10 gpu
   ```
## Example Output
```
Grid Sizes: 256 256 256
Memory Usage: 230 MB
 ***** Running C++ Serial variant *****
Initializing ...
--------------------------------------
time         : 2.92984 secs
throughput   : 57.2632 Mpts/s
flops        : 3.49306 GFlops
bytes        : 0.687159 GBytes/s

--------------------------------------

--------------------------------------
 ***** Running SYCL variant *****
Initializing ...
 Running on Intel(R) Gen9
 The Device Max Work Group Size is : 256
 The Device Max EUCount is : 48
 The blockSize x is : 32
 The blockSize y is : 8
 Using Global Memory Kernel
--------------------------------------
time         : 0.597494 secs
throughput   : 280.793 Mpts/s
flops        : 17.1284 GFlops
bytes        : 3.36952 GBytes/s

--------------------------------------

--------------------------------------
Final wavefields from SYCL device and CPU are equivalent: Success
--------------------------------------
```

## License
Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
