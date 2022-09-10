 # `All Pairs Shortest Paths` Sample

The `All Pairs Shortest Paths` sample uses the Floyd-Warshall algorithm to find the
shortest paths between pairs of vertices in a graph. The code in the sample uses a parallel blocked algorithm that enables the application to offload compute intensive work to the GPU efficiently.

| Property                          | Description
|:---                               |:---
| What you will learn               | How to offload compute intensive parts of the application using lambda kernel. <br> How to measure kernel execution time.
| Time to complete                  | 15 minutes

## Purpose
This sample uses the blocked Floyd-Warshall all pairs shortest paths algorithm to
compute a matrix representing the minimum distance from any node to all other
nodes in the graph. 

Using parallel blocked processing, blocks can be calculated simultaneously by distributing task computations to the GPU. The application executes sequentially and in parallel with runtimes both displayed in the output, which allows you to compare the difference. 

The parallel implementation of the blocked Floyd-Warshall algorithm has three phases. Given that a prior round of these computation phases completes, the following is true:

- Phase 1 is independent.
- Phase 2 executes only after phase 1 completes.
- Phase 3 depends on phase 2, and executes only after phase 2 completes.

The inner loop of the sequential implementation is: `g[i][j] = min(g[i][j], g[i][k] + g[k][j])`. 
Careful observation shows that for the kth iteration of the outer loop, the
computation depends on cells either on the kth column, `g[i][k]` or on the kth
row, `g[k][j]` of the graph. The phases occur in the following sequence:

- Phase 1 handles `g[k][k]`
- Phase 2 handles `g[\*][k]` and `g[k][\*]`
- Phase 3 handles `g[\*][\*]`

These cell level observations largely propagate to the blocks as well. In each phase, computation within a block can proceed independently in parallel.

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
Key SYCL* concepts demonstrated in the code sample include using device selector, unified shared memory, kernel, and command groups to implement a solution using a parallel block method targeting the GPU.

For comprehensive information about oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

## Setting Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `All Pairs Shortest Paths` Program for CPU and GPU
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

> **Note**: You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

> **Note**: To tune the list of components and the version of those components, use a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

### Running Samples in DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and run in batch or interactive mode. 

For specific instructions, jump to [Run the sample in the DevCloud](#run-on-devcloud)

For more information, see the [Intel&reg; oneAPI Base Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### Include Files
The include folder is at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system. You might need to use some of the resources from this location to build the sample.

### Using Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
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
If an error occurs, you can get more details by running `make` with `VERBOSE=1`:
```
make VERBOSE=1
```
### On Windows*
**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
2. Right-click on the project in **Solution Explorer** and select **Rebuild**.

**Using MSBuild**

1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command:
   ```
   MSBuild all-pairs-shortest-paths.sln /t:Rebuild /p:Configuration="Release"
   ```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `All Pairs Shortest Paths` Program
### On Linux

1. Run the program.
   ```
   make run
   ```
### On Windows
 1. Change to the output directory.
 2. Run the executable.
    ```
    all-pairs-shortest-paths.exe
    ```

### Run the `All Pairs Shortest Paths` Sample in Intel® DevCloud
If running a sample in the Intel® DevCloud, you must specify the compute node
(CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more
information, see the Intel® oneAPI Base Toolkit [Get Started
Guide](https://devcloud.intel.com/oneapi/get_started/).

### Example Output
The output displays the device on which the program ran.
```
Device: Intel(R) Gen9
Repeating computation 8 times to measure run time ...
Iteration: 1
Iteration: 2
Iteration: 3
Iteration: 4
Iteration: 5
Iteration: 6
Iteration: 7
Iteration: 8
Successfully computed all pairs shortest paths in parallel!
Time sequential: 0.583029 sec
Time parallel: 0.159223 sec
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

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).