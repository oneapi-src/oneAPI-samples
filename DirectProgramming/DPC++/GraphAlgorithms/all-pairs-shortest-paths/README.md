 # `All Pairs Shortest Paths` Sample

`All Pairs Shortest Paths` uses the Floyd-Warshall algorithm to find the
shortest paths between pairs of vertices in a graph. It uses a parallel blocked
algorithm that enables the application to offload compute intensive work to the
GPU efficiently.

For comprehensive instructions, see the [Intel&reg; oneAPI Programming
Guide](https://software.intel.com/en-us/oneapi-programming-guide) and search
based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br>Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | Offloading compute intensive parts of the application using lambda kernel <br>Measuring kernel execution time
| Time to complete                  | 15 minutes


## Purpose

This sample uses blocked Floyd-Warshall all pairs shortest paths algorithm to
compute a matrix representing the minimum distance from any node to all other
nodes in the graph. Using parallel blocked processing, blocks can be calculated
simultaneously by distributing task computations to the GPU. For comparison, the
application is run sequentially and in parallel with run times for displayed
in each application output. The device where the code is run is also
identified.

The parallel implementation of the blocked Floyd-Warshall algorithm has three
phases. Given that a prior round of these computation phases is complete, phase
1 is independent; Phase 2 can only execute after phase 1 completes; Similarly,
phase 3 depends on phase 2 to only execute after phase 2 is complete.

The inner loop of the sequential implementation is: $g[i][j] = min(g[i][j],
  g[i][k] + g[k][j])$

A careful observation shows that for the kth iteration of the outer loop, the
computation depends on cells either on the kth column, g[i][k] or on the kth
row, g[k][j] of the graph. Phase 1 handles g[k][k], phase 2 handles g[\*][k] and
g[k][\*], and phase 3 handles g[\*][\*] in that sequence. These cell level
observations largely propagate to the blocks as well.

In each phase, computation within a block can proceed independently in parallel.


## Key implementation details
Includes device selector, unified shared memory, kernel, and command groups to
implement a solution using a parallel block method targeting the GPU.


## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).


## Setting Environment Variables

For working at a Command-Line Interface (CLI), the tools in the oneAPI toolkits
are configured using environment variables. Set up your CLI environment by
sourcing the ``setvars`` script every time you open a new terminal window. This
will ensure that your compiler, libraries, and tools are ready for development.

### Linux*
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

>**Note**: [Modulefiles
    scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html)
    can also be used to set up your development environment. The modulefiles
    scripts work with all Linux shells.

> **Note**: If you wish to fine tune the list of components and the version of
    those components, use a [setvars config
    file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html)
    to set up your development environment.

### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics
Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find
missing dependencies and permissions errors. See [Diagnostics Utility for
Intel&reg; oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### Windows*

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

## Building the Program for CPU and GPU

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
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on
your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, you must specify the compute node
(CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more
information, see the Intel&reg; oneAPI Base Toolkit [Get Started
Guide](https://devcloud.intel.com/oneapi/get_started/).

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
Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel&reg; oneAPI Toolkits, return
to this readme for instructions on how to build and run a sample.

### On Linux

Perform the following steps:

1.  Build the program using the following <code> cmake </code> commands.
```
    $ cd all-pairs-shortest-paths
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

2.  Run the program.
```
    $ make run
```

If an error occurs, you can get more details by running `make` with the
`VERBOSE=1` argument: ``make VERBOSE=1`` For more comprehensive troubleshooting,
use the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors. [Learn
more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

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
      MSBuild all-pairs-shortest-paths.sln /t:Rebuild /p:Configuration="Release"
     ```

## Running the sample

### Example Output
```
Device: Intel(R) Gen9
Repeating computation 8 times to measure run time ...
Iteration: 1
Iteration: 2
Iteration: 3
...
Iteration: 8
Successfully computed all pairs shortest paths in parallel!
Time sequential: 0.583029 sec
Time parallel: 0.159223 sec
```