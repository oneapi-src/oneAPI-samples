# `Nbody` sample
An N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity. This `Nbody` sample code uses SYCL* standards for CPUs and GPUs.

| Property             | Description
|:---                  |:---
| What you will learn  | How to offload complex computations to GPU
| Time to complete     | 15 minutes

## Purpose
By default, the `Nbody` sample code simulates 16,000 particles over ten integration steps. The position, velocity, and acceleration parameters of each particle are dependent on other (N-1) particles.

This algorithm is highly data parallel, so the code a good candidate to offload to GPU. The code demonstrates how to deal with multiple device kernels, which can be enqueued into a SYCL queue for execution, and the code demonstrates how to handle parallel reductions.

## Prerequisites
| Optimized for     | Description
|:---               |:---
| OS                | Ubuntu* 18.04 <br> Windows* 10
| Hardware          | Skylake with GEN9 or newer
| Software          | Intel® oneAPI DPC++ Compiler

## Key Implementation Details
The basic SYCL* compliant implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## Build the `Nbody` Program for CPU and GPU

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
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
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
1. Build the program.
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

### On Windows*
**Using Visual Studio**

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. From the top menu, select **Debug** > **Start without Debugging**.

**Using MSBuild**

1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild Nbody.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Nbody` Program
### On Linux
1. Run the program.
   ```
   make run
   ```
2. Clean the program. (Optional)
   ```
   make clean
   ```
### On Windows
1. Change to the output directory.
2. Run the executable.
   ```
   Nbody.exe
   ```

## Example Output
### Application Parameters
You can modify the `NBody` sample simulation parameters in `GSimulation.cpp`. Configurable parameters include:

|Parameter       | Defaults
|:---            |:---
|`set_npart`     | Default number of particles is **16000**
|`set_nsteps`    | Default number of integration steps is **10**
|`set_tstep`     | Default time delta is **0.1**
|`set_sfreq`     | Default sample frequency is **1**

### Example Output on Linux
```
===============================
Initialize Gravity Simulation
Target Device: Intel(R) Gen9
nPart = 16000; nSteps = 10; dt = 0.1
------------------------------------------------
s       dt      kenergy     time (s)    GFLOPS
------------------------------------------------
1       0.1     26.405      0.28029     26.488
2       0.2     313.77      0.066867    111.03
3       0.3     926.56      0.065832    112.78
4       0.4     1866.4      0.066153    112.23
5       0.5     3135.6      0.065607    113.16
6       0.6     4737.6      0.066544    111.57
7       0.7     6676.6      0.066403    111.81
8       0.8     8957.7      0.066365    111.87
9       0.9     11587       0.066617    111.45
10      1       14572       0.06637     111.86

# Total Time (s)     : 0.87714
# Average Performance : 112.09 +- 0.56002
===============================
Built target run
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
