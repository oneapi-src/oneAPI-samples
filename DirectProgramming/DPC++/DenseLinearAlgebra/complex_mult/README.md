# `Complex Multiplication` Sample

complex multiplication is a program that multiplies two large vectors of
Complex numbers in parallel and verifies the results. It also implements
a custom device selector to target a specific vendor device. This program is
implemented using C++ and DPC++ language for Intel CPU and accelerators.
The Complex class is a custom class, and this program shows how we can use
custom types of classes in a DPC++ program.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | Using custom type classes and offloads complex number computations to GPU using DPC++
| Time to complete                  | 15 minutes


## Purpose

Complex multiplication multiplies two vectors with complex numbers. The
code will attempt to run the calculation on both the GPU and CPU and then
verifies the results. The size of the computation can be adjusted for
heavier workloads. If successful, the name of the offload device and a
success message is displayed.

This sample uses buffers to manage memory. For more information regarding
different memory management options, refer to the vector_add sample.

Complex multiplication includes both C++ and DPC++ implementations.

This program shows how to create a custom device selector and to target
GPU or CPU of a specific vendor. The program also shows how to pass in a
vector of custom Complex class objects to do the parallel
executions on the device. The device used for the compilation is displayed in
the output.


## Key implementation details

This program shows how we can use custom types of classes in a DPC++
program and explains the basic DPC++ implementation, including device
selector, buffer, accessor, kernel and command group.


## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the complex_mult Program for CPU and GPU

Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify
the compute node (CPU, GPU, FPGA) and whether to run in batch or
interactive mode.

For specific instructions, jump to [Run the sample in the DevCloud](#run-on-devcloud)

For more information, see the Intel® oneAPI Base Toolkit
[Get Started Guide](https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### Using Visual Studio Code\*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to
this readme for instructions on how to build and run a sample.


### Setting Environment Variables


For working at a Command-Line Interface (CLI), the tools in the oneAPI toolkits
are configured using environment variables. Set up your CLI environment by
sourcing the ``setvars`` script every time you open a new terminal window. This
will ensure that your compiler, libraries, and tools are ready for development.


#### Linux
Source the script from the installation location, which is typically in one of
these folders:


For root or sudo installations:


  ``. /opt/intel/oneapi/setvars.sh``


For normal user installations:

  ``. ~/intel/oneapi/setvars.sh``

**Note:** If you are using a non-POSIX shell, such as csh, use the following command:

     ``$ bash -c 'source <install-dir>/setvars.sh ; exec csh'``

If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


**Note:** [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html)
    can also be used to set up your development environment.
    The modulefiles scripts work with all Linux shells.


**Note:** If you wish to fine
    tune the list of components and the version of those components, use
    a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html)
    to set up your development environment.

#### Windows

Execute the  ``setvars.bat``  script from the root folder of your
oneAPI installation, which is typically:


  ``"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"``


For Windows PowerShell* users, execute this command:

  ``cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'``


If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### Set Environment Variables

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### On a Linux\* System
   * Build the program using Make
    make all

   * Run the program
    make run

   * Clean the program
    make clean

### On a Windows\* System Using Visual Studio\* Version 2017 or Newer

- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild complex_mult.sln /t:Rebuild /p:Configuration="debug"`

## Running the Sample

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

### Running the sample in the DevCloud<a name="run-on-devcloud"></a>

#### Build and run

To launch build and run jobs on DevCloud submit scripts to PBS through the qsub utility.
> Note that all parameters are already specified in the build and run scripts.

1. Build the sample on a gpu node.

    ```bash
    qsub build.sh
    ```

2. When the build job completes, there will be a `build.sh.oXXXXXX` file in the directory. After the build job completes, run the sample on a gpu node:

    ```bash
    qsub run.sh
    ```

#### Additional information

1. In order to inspect the job progress, use the qstat utility.

    ```bash
    watch -n 1 qstat -n -1
    ```

    > Note: The watch `-n 1` command is used to run `qstat -n -1` and display its results every second.

2. When a job terminates, a couple of files are written to the disk:

    <script_name>.sh.eXXXX, which is the job stderr

    <script_name>.sh.oXXXX, which is the job stdout

    > Here XXXX is the job ID, which gets printed to the screen after each qsub command.

3. To inspect the output of the sample use cat command.

    ```bash
    cat run.sh.oXXXX
    ```

### Build and run additional samples

Several sample programs are available for you to try, many of which can be
compiled and run in a similar fashion to this sample. Experiment with running
the various samples on different kinds of compute nodes or adjust their source
code to experiment with different workloads.

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)