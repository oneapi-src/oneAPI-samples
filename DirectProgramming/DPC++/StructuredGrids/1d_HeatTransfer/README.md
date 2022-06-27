# `1D-Heat-Transfer` Sample

This code sample demonstrates the simulation of a one-dimensional heat transfer process using SYCL*. Kernels in this example are implemented as a discretized differential equation with the second derivative in space and the first derivative in time.

For comprehensive information in using oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide), and use search or the table of contents to find relevant information.

| Property                     | Description
|:---                               |:---
| What you will learn               | How to simulate 1D Heat Transfer using Intel&reg; oneAPI DPC++ Compiler
| Time to complete                  | 10 minutes


## Purpose

1D-Heat-Transfer is a SYCL*-compliant application that simulates the heat
propagation on a one-dimensional isotropic and homogeneous medium. The
following equation is used in the simulation of heat propagation:

$dU/dt = k * d^2U/dx^2$

Where:
- $dU/dt$ is the rate of change of temperature at a point.
- $k$ is the thermal diffusivity.
- $d^2U/dx^2$ is the second spatial derivative.

or

$U(i) = C * (U(i+1) - 2 * U(i) + U(i-1)) + U(i)$

Where:
- constant $C = k * dt / (dx * dx)$

The code sample includes both parallel and serial calculations of heat
propagation. The code sample attempts to execute on an
available GPU and will fallback to the system CPU if a compatible GPU is
not detected. The results are stored in a file.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler

## Key Implementation Details

The basic SYCL* implementation explained in the code includes a device
selector, buffer, accessor, USM allocation, kernel, and command
groups.

## Build the 1d_HeatTransfer Program for CPU and GPU

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

The include folder is located at
`%ONEAPI_ROOT%\dev-utilities\latest\include` on your development
system".


### Use Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel&reg; oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


### On Linux* 
1. Build the program using the following `cmake` commands.
   ```
   $ cd 1d_HeatTransfer
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make -j
   ```
2. Run the program
   ```
   $ make run
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

- Build the program using MSBuild.
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild 1d_HeatTransfer.sln /t:Rebuild /p:Configuration="Release"`

### Run Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

## Run the Sample
### Application Parameters

Usage: `1d_HeatTransfer <n> <i>`

Where:
- `n` is the number of points you want to simulate the heat transfer.
- `i` is the number of timesteps in the simulation.

  To simulate 100 points for 1000 timesteps: `./1d_HeatTransfer 100 1000`

The sample performs the computation serially on CPU, using SYCL* buffers and USM. The results are compared against the serial version and saved to `usm_error_diff.txt` and
`buffer_error_diff.txt`.  If the results match, the application will
display a `PASSED!` message.

### Example of Output
```
$ make run
[100%] Built target 1d_HeatTransfer
Number of points: 100
Number of iterations: 1000
Using buffers
  Kernel runs on Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz
  Elapsed time: 0.439238 sec
  PASSED!
Using USM
  Kernel runs on Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz
  Elapsed time: 0.193435 sec
  PASSED!
[100%] Built target run
$
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).