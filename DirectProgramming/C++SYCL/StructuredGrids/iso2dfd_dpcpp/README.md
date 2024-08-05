# `ISO2DFD` Sample
The `ISO2DFD` sample demonstrates two-dimensional finite-difference wave propagation in isotropic media. The sample is a two-dimensional stencil to simulate a wave propagating in a 2D isotropic medium, and the code illustrates the basics of SYCL* code using direct programming.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to offload complex computation to a GPU
| Time to complete         | 10 minutes

## Purpose
ISO2DFD is a finite difference stencil kernel for solving the 2D acoustic isotropic wave equation.  The sample uses a Partial Differential Equation (PDE), with a finite-difference method, to illustrate the essential elements of SYCL* queues, buffers, accessors, and kernels.

> **Note**: You can find a complete code walk-through of this sample at [Code Sample: Two-Dimensional Finite-Difference Wave Propagation in Isotropic Media (ISO2DFD)](https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd).

You can use this sample code as an entry point to start SYCL* programming or as a proxy to develop or better understand complicated code for similar problems.

## Prerequisites
| Optimized for            | Description
|:---                      |:---
| OS                       | Ubuntu* 18.04 <br> Windows* 10
| Hardware                 | Skylake with GEN9 or newer
| Software                 | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The sample demonstrates several SYCL implementations.
- SYCL queues (including device selectors and exception handlers).
- SYCL buffers and accessors.
- The ability to call a function inside a kernel definition and pass accessor arguments as pointers. A function called inside the kernel performs a computation (it updates a grid point specified by the global ID variable) for a single time step.

The sample runs on the GPU and CPU to calculate a result. The results from the two devices are compared. If the sample ran correctly, the program reports success.

The output includes the GPU device name.

##  Build the `ISO2DFD` Program for CPU and GPU

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
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
1. Change to the sample directory.
1. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make -j
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
3. Run the following command: `MSBuild iso2dfd.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `ISO2DFD` Sample
### Application Parameters
The program requires grid size and time steps to execute.
```
program <n1> <n2> <iterations>
```
where:
| Parameter        | Description
|:---              |:---
|`n1 n2`           | Grid size for the stencil. `n1` is X (rows) and `n2` is Y (columns). Use `n1` = **1000** and `n2` = **1000** for results that match the output below.
| `iterations`     |Number of timesteps. Use `iterations` = **2000** for results that match the output below.

To specify a grid size of 1000x1000 and 2000 time steps iterations, you would use the following command: `iso2dfd 1000 1000 2000`.

### On Linux
1. Run the program.
    ```
    make run
    ```
2. Clean the project files. (Optional)
   ```
   make clean
   ```
### On Windows
1. Change to the output directory.
2. Specify the input parameters, and run the program.
   ```
   iso2dfd.exe 1000 1000 2000
   ```

### Example Output
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
```

If you run the program on Linux, you can use the generated .bin files to plot the wave field output using the SU seismic processing library, which is part of the Seismic Un*x Package. The library contains utilities to display seismic wave fields, and the processing library is available from [John Stockwell's SeisUnix](https://github.com/JohnWStockwellJr/SeisUnix) GitHub Repository.

You can find graphical output examples for the program at [Code Sample: Two-Dimensional Finite-Difference Wave Propagation in Isotropic Media (ISO2DFD)](https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
