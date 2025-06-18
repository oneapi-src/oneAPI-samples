# `1D-Heat-Transfer` Sample

This `1D-Heat-Transfer` sample demonstrates the simulation of a one-dimensional heat transfer process. Kernels in this example are implemented as a discretized differential equation with the second derivative in space and the first derivative in time.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to simulate 1D Heat Transfer using SYCL*
| Time to complete         | 10 minutes


## Purpose
This `1D-Heat-Transfer` sample is an application that simulates the heat propagation on a one-dimensional isotropic and homogeneous medium. The code sample includes both parallel and serial calculations of heat propagation.

The following equation is used in the simulation of heat propagation:

$
dU/dt = k * d^2U/dx^2
$

where:
- $dU/dt$ is the rate of change of temperature at a point.
- k is the thermal diffusivity.
- $d^2U/dx^2$ is the second spatial derivative.

or:

$
U(i) = C * (U(i+1) - 2 * U(i) + U(i-1)) + U(i)
$

where:
- constant $C = k * dt / (dx * dx)$

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic SYCL* implementation explained in the code includes a device selector, buffer, accessor, USM allocation, kernel, and command groups.

The program attempts to offload the computations to a GPU first. If the program cannot detect a compatible GPU, the program runs on the CPU (host device).

## Build the `1D-Heat-Transfer` Program for CPU and GPU

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
   make -j
   ```
> **Note**: The `make -j` flag allows multiple jobs simultaneously.

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
3. Run the following command: `MSBuild 1d_HeatTransfer.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `1D-Heat-Transfer` Program
### Application Parameters
The program requires two inputs. General usage syntax is as follows:

`1d_HeatTransfer <n> <i>`

| Input         | Description
|:---           |:---
| `n`           | The number of points you want to simulate the heat transfer.
| `i`           | The number of timesteps in the simulation.

The sample performs the computation serially on CPU using buffers and USM. The parallel results are compared to serial version. The output of the comparisons is saved to `usm_error_diff.txt` and
`buffer_error_diff.txt` in the output directory. If the results match, the application will
display a `PASSED!` message.

### On Linux
1. Run the program
   ```
   $ make run
   ```
2. Clean project files. (Optional)
   ```
   make clean
   ```

### On Windows
1. Change to the output directory.
2. Specify the input values, and run the program.
   ```
   1d_HeatTransfer 100 1000
   ```

## Example Output
```
Number of points: 100
Number of iterations: 1000
Using buffers
  Kernel runs on Intel(R) UHD Graphics P630 [0x3e96]
  Elapsed time: 0.506778 sec
  PASSED!
Using USM
  Kernel runs on Intel(R) UHD Graphics P630 [0x3e96]
  Elapsed time: 0.00926689 sec
  PASSED!
```

The parallel to serial comparisons are saved to `usm_error_diff.txt` and `buffer_error_diff.txt` in the output directory.

## License
Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
