# `Bitonic Sort` Sample

This code sample demonstrates the implementation of bitonic sort using Data
Parallel C++ to offload the computation to a GPU. In this implementation, a
random sequence of 2**n elements is given (n is a positive number) as input, and
the algorithm sorts the sequence in parallel. The result sequence is in
ascending order.

For comprehensive instructions, see the [Intel&reg; oneAPI Programming
Guide](https://software.intel.com/en-us/oneapi-programming-guide) and search
based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | Implement bitonic sort using Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes


## Purpose

The algorithm converts a randomized sequence of numbers into a bitonic sequence
(two ordered sequences) and then merges these two ordered sequences into an
ordered sequence. Bitonic sort algorithm is briefly described as followed:

- First, it decomposes the randomized sequence of size 2\*\*n into 2\*\*(n-1)
pairs where each pair consists of 2 consecutive elements. Note that each pair is
a bitonic sequence.
- Step 0: for each pair (sequence of size 2), the two elements are swapped so
that the two consecutive pairs form  a bitonic sequence in increasing order, the
next two pairs form the second bitonic sequence in decreasing order. The next
two pairs form the third bitonic sequence in increasing order, etc., .... At the
end of this step, we have 2\*\*(n-1) bitonic sequences of size 2, and they
follow an order of increasing, decreasing, increasing, .., decreasing. Thus,
they form 2\*\*(n-2) bitonic sequences of size 4.
- Step 1: for each new 2\*\*(n-2) bitonic sequences of size 4, (each new
sequence consists of 2 consecutive previous sequences), it swaps the elements so
that at the end of step 1, we have 2\*\*(n-2) bitonic sequences of size 4, and
they follow an order: increasing, decreasing, increasing, ..., decreasing. Thus,
they form 2\*\*(n-3) bitonic sequences of size 8.
- Same logic applies until we reach the last step.
- Step n: at this last step, we have one bitonic sequence of size 2\*\*n. The
elements in the sequence are swapped until we have a sequence in increasing
order.

The code attempts to execute on an available GPU and fallback to the system CPU
if a compatible GPU is not detected.

## Key Implementation Details

The basic SYCL* implementation explained in the code includes device selector,
buffer, accessor, kernel, and command groups. Unified Shared Memory (USM) and
Buffer Object are used for data management.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI
   Toolkits**.
 - Configure the oneAPI environment with the extension **Environment
   Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment,
see the [Using Visual Studio Code with Intel&reg; oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel&reg; oneAPI Toolkits, return
to this readme for instructions on how to build and run a sample.

## Setting Environment Variables

For working at a Command-Line Interface (CLI), the tools in the oneAPI toolkits
are configured using environment variables. Set up your CLI environment by
sourcing the ``setvars`` script every time you open a new terminal window. This
will ensure that your compiler, libraries, and tools are ready for development.

### Linux
Source the script from the installation location, which is typically in one of
these folders:

For system wide installations:

  ``. /opt/intel/oneapi/setvars.sh``

For private installations:

  ``. ~/intel/oneapi/setvars.sh``

>**Note**: If you are using a non-POSIX shell, such as csh, use the following
>command:
  ```
    $ bash -c 'source <install-dir>/setvars.sh ; exec csh'
  ```
If environment variables are set correctly, you will see a confirmation message.

>**Note:** [Modulefiles
>scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html)
>can also be used to set up your development environment. The modulefiles
>scripts work with all Linux shells.

> **Note:** If you wish to fine tune the list of components and the version of
    those components, use a [setvars config
    file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html)
    to set up your development environment.

### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics
Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find
missing dependencies and permissions errors. See [Diagnostics Utility for
Intel&reg; oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### Windows

Execute the  ``setvars.bat``  script from the root folder of your oneAPI
installation, which is typically:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
For Windows PowerShell* users, execute this command:
```
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
```
If environment variables are set correctly, you will see a confirmation message.

## Building the `bitonic-sort` Program for CPU and GPU

> **Note**: If you have not already done so, set up your CLI environment by
> sourcing  the `setvars` script located in the root of your oneAPI
> installation.
>
> Linux:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for
>[Linux or
>macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html),
>or
>[Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Include Files
The include folder is located at ``%ONEAPI_ROOT%\dev-utilities\latest\include``
on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, you must specify the compute node
(CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more
information, see the Intel&reg; oneAPI Base Toolkit [Get Started
Guide](https://devcloud.intel.com/oneapi/get_started/).

### On Linux
1. Build the program using the following `cmake` commands.
    ```
    $ cd bitonic-sort
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```

2. Run the program:
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```

If an error occurs, you can get more details by running `make` with the
`VERBOSE=1` argument: ``` make VERBOSE=1 ```
### On Windows Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019
      IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select **Debug -> Start without Debugging**.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools
       Command Prompt for VS2019"
     - Run the following command:
     ```
     MSBuild bitonic-sort.sln /t:Rebuild /p:Configuration="Release"
     ```

## Running the sample
### Application Parameters

Usage: `bitonic-sort <exponent> <seed>`

where:

- exponent is a positive number. The according length of the sequence is
  2**exponent.
- seed is the seed used by the random generator to generate the randomness.

The sample offloads the computation to GPU and then performs the computation in
serial in the CPU. The results from the parallel and serial computation are
compared. If the results are matched and the ascending order is verified, the
application will display a “Success!” message.

### Example of Output
```
$ ./bitonic-sort 21 47
Array size: 2097152, seed: 47
Device: Intel(R) Gen9 HD Graphics NEO
Warm up ...
Kernel time using USM: 0.248422 sec
Kernel time using buffer allocation: 0.253364 sec
CPU serial time: 0.628803 sec

Success!
```
