# `Loop Unroll` Sample

The `Loop Unroll` sample demonstrates a simple method of unrolling loops using SYCL* to improve throughput of a program for GPU offload.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to perform loop unrolling using SYCL
| Time to complete      | 30 minutes

## Purpose

The concepts demonstrated by the sample are:

- Understanding basics of loop unrolling.
- Unrolling loops in your program.
- Determining optimal unroll factor for your program.

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS	                 | Ubuntu* 18.04 <br> Windows* 10
| Hardware	           | Skylake with GEN9 or newer
| Software	           | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

The loop unrolling mechanism is used to increase program parallelism by duplicating the compute logic within a loop. The **unroll factor** refers to the number of times the loop logic is duplicated. Depending on whether the **unroll factor** is equal to the number of loop iterations or not, loop unroll methods can be categorized as full-loop unrolling and partial-loop unrolling. A full unroll is a special case where the **unroll factor** is equal to the number of loop iterations.

The code in this sample will attempt to run on a compatible GPU. If the program cannot find a compatible GPU, it will fall back to the CPU.

## Build the `Loop Unroll` Sample

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

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the Linux* instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

1. Change to the sample directory.
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

**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. Right-click on the project in **Solution Explorer** and select **Rebuild**.
3. From the top menu, select **Debug** > **Start without Debugging**.

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild loop-unroll.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run `Loop Unroll` the Sample

>**Note**: A workload that takes more than four seconds for GPU hardware to execute is a long running workload. By default, individual threads that qualify as long-running workloads are considered hung and are terminated. By disabling the hangcheck timeout period, you can avoid this problem.
>
> To prevent long GPU compute kernel workloads from being terminated, disable hangcheck.
>
> - For Linux, see the *GPU: Disable Hangcheck* section in the [*Get Started with the Intel® oneAPI Base Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-base-linux/top/before-you-begin.html) document.
>
> - For Windows, see the **Long-running applications are terminated** row in the *Troubleshooting* topic of the [*Get Started with the Intel® oneAPI Base Toolkit for Windows**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-base-windows/top/troubleshooting.html) document.

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
2. Run the program.
   ```
   loop-unroll.exe
   ```

#### Build and Run the Sample

Follow the Linux instructions to build and run the program.

## Example Output

```
Input array size: 67108864
Running on device: Intel(R) Gen9
Unroll factor: 1 Kernel time: 13710.9 ms
Throughput for kernel with unroll factor 1: 0.005 GFlops
Unroll factor: 2 Kernel time: 8906.831 ms
Throughput for kernel with unroll factor 2: 0.008 GFlops
Unroll factor: 4 Kernel time: 4661.967 ms
Throughput for kernel with unroll factor 4: 0.014 GFlops
Unroll factor: 8 Kernel time: 2669.343 ms
Throughput for kernel with unroll factor 8: 0.025 GFlops
Unroll factor: 16 Kernel time: 2421.305 ms
Throughput for kernel with unroll factor 16: 0.028 GFlops
PASSED: The results are correct.
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
