# `Complex Multiplication` Sample

The `Complex Multiplication` sample demonstrates how to multiply two large vectors of complex numbers in parallel and verifies the results. The program also implements
a custom device selector to target a specific vendor device. This program is
implemented using SYCL* and C++ for Intel CPU and accelerators.



| Property                     | Description
|:---                               |:---
| What you will learn               | Using custom type classes and offloads complex number computations to GPU using SYCL*
| Time to complete                  | 15 minutes


## Purpose

This sample program shows how to create a custom device selector, how to target GPU or CPU of a specific vendor, and how to pass in a
vector of custom Complex class objects to do the parallel
executions on the device. The device used for the compilation is displayed in
the output.

Complex multiplication multiplies two vectors with complex numbers. The
code will attempt to run the calculation on both the GPU and CPU and then
verifies the results. The size of the computation can be adjusted for
heavier workloads. If successful, the program displays the name of the offload device and along with a success message. (This sample uses buffers to manage memory. (For more information regarding different memory management options, refer to the [`Vector Add`](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/DenseLinearAlgebra/vector-add) sample.)

This `Complex Multiplication` sample includes both C++ and SYCL* implementations.



## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br>Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler

## Key implementation details

This program shows how we can use custom types of classes in a SYCL*-compliant program and explains basic implementation, including device selector, buffer, accessor, kernel and command group.

The `Complex` class, described in `Complex.hpp` is a custom class, and this program shows how we can use custom types of classes in a SYCL*-compliant program.


## Build the `Complex Multiplication` Sample for CPU and GPU

### Setting Environment Variables
For working with the Command-Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
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
>For more information on environment variables, see "Use the setvars Script" for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

If you wish to fine tune the list of components and the version of those components, use
a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

### Include Files
The include folder is located on your development system at `%ONEAPI_ROOT%\dev-utilities\latest\include`.

### Using Visual Studio Code*  (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
1. Build the program.
   ```
   make all
   ```

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
make VERBOSE=1
```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### On Windows* Using Visual Studio*
- Build the program using VS2017 or later.
    - Right-click on the solution file and open using the IDE.
    - Right-click on the project in **Solution Explorer** and select **Rebuild**.
    - From the top menu, select **Debug** > **Start without Debugging**.

- Build the program using MSBuild.
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio version.
     - Run the following command: `MSBuild complex_mult.sln /t:Rebuild /p:Configuration="debug"`


## Run the `Complex Multiplication`  Sample
### Application Parameters
There are no editable parameters for this sample.

## Example of Output

```
	Target Device: Intel(R) Gen9

	****************************************Multiplying Complex numbers in Parallel********************************************************
	[0] (2 : 4i) * (4 : 6i) = (-16 : 28i)
	[1] (3 : 5i) * (5 : 7i) = (-20 : 46i)
	[2] (4 : 6i) * (6 : 8i) = (-24 : 68i)
	[3] (5 : 7i) * (7 : 9i) = (-28 : 94i)
	[4] (6 : 8i) * (8 : 10i) = (-32 : 124i)
	...
	[9999] (10001 : 10003i) * (10003 : 10005i) = (-40012 : 200120014i)
	Complex multiplication successfully run on the device

```
## Build the `Complex Multiplication` Sample in Intel&reg; DevCloud

When running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

To launch build and run jobs on Intel&reg; DevCloud submit scripts to Portable Bash Script (PBS) through the qsub utility.

> **Note**: All parameters are already specified in the build and run scripts.

1. Open a terminal on a Linux system.

2. Log in to the Intel&reg; DevCloud.
   ```
   ssh devcloud
   ```

3. Download the samples from GitHub.
   ```
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```
4. Change directories to the sample directory.

5. Build the sample on a gpu node.

    ```
    qsub build.sh
    ```
6. Use the qstat utility to inspect the job progress.
   ```
   watch -n 1 qstat -n -1
   ```
   - The watch `-n 1` command is used to run `qstat -n -1` and display its results every second.

When the build job completes, there will be a **build.sh.oXXXXXX** file in the directory.

> **Note**: Some files are written to the disk when each job terminates.
> - **<script_name>.sh.eXXXX** = the job stderr
> - **<script_name>.sh.oXXXX** = the job stdout
> - where **XXXX** is the job ID printed to the screen after each qsub command.


7. After the build job completes, run the sample on a gpu node:

    ```
    qsub run.sh
    ```

3. Inspect the output of the sample using the `cat` command.

    ```
    cat run.sh.oXXXX
    ```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).