# `Hidden Markov Models` Sample

The `Hidden Markov Models` sample presents a statistical model using a Markov process to present graphable nodes that are otherwise in an unobservable state or “hidden”. This technique helps with pattern recognition such as speech, handwriting, gesture recognition, parts-of-speech tagging, partial discharges, and bioinformatics. The sample offloads the complexity of the Markov process to the GPU.

| Property                     | Description
|:---                          |:---
| What you will learn          | Implement Viterbi algorithm to get the most likely path that consists of the hidden states
| Time to complete             | 10 minutes

## Purpose
The sample can use GPU offload to compute sequential steps of multiple graph
traversals simultaneously.

The directed edges of this graph are possible transitions between nodes or states defined with the following parameters:
- the number of states is N
- the transition matrix A is a square matrix of size N
- Each element with indexes (i,j) of this matrix determines the probability to move from the state i to the state j on any step of the Markov process (i and
  j can be the same if the state does not change on the taken step)

The main assumption in the method is that there are visible observations that depend on the current Markov process. The dependency can be described as a conditional probability distribution (represented by emission matrix). The problem is to find out the most likely chain of the hidden Markov states using the given observations set.

This code sample implements the Viterbi algorithm, a dynamic programming algorithm for finding the most likely sequence of hidden states—called the
Viterbi path—that results in a sequence of observed events, especially in Markov information sources and HMM.

- Initially, the dataset for algorithm processing is generated: initial states probability distribution Pi, transition matrix A, emission matrix B, and the sequence or the observations produced by hidden Markov process.
- First, the matrix of Viterbi values on the first states is initialized using distribution Pi and emission matrix B. The matrix of back pointers is initialized with default values -1.
- Then, for each time step, the Viterbi matrix is set to the maximal possible value using A, B, and Pi.
- Finally, the state with maximum Viterbi value on the last step is set as a path to the Viterbi final state. The previous nodes of this path are determined using the back pointers matrix correspondent rows for each step except the last one.

> **Note**: The implementation uses logarithms of the probabilities to process
> small numbers correctly and replace multiplication operations with addition
> operations.

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++ Compiler

## Key Implementation Details
The basic SYCL* implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## Build the `Hidden Markov Models` Program for CPU and GPU

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

### Using Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment
    Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel®
    oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment,
see the [Using Visual Studio Code with Intel® oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
1. Change to the sample directory.
2. Build the program.
    ```
    mkdir build
    cd build
    cmake ..
    make
    ```

If an error occurs, you can get more details by running `make` with the
`VERBOSE=1` argument:
```
make VERBOSE=1
```

### On Windows*
**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. From the top menu, select **Debug** -> **Start without Debugging**.

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools
   Command Prompt for VS2019" or whatever is appropriate for your Visual Studio*
   version.
2. Change to the sample directory.
3. Run the following command:
   ```
   MSBuild hidden-markov-models.sln /t:Rebuild /p:Configuration="Release"
   ```

### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics
Utility for Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel® oneAPI
Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Run the `Hidden Markov Models` Sample
### On Linux
1. Run the program:
    ```
    make run
    ```
2. Clean the program. (Optional)
    ```
    make clean
    ```

### On Windows
Use **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. From the top menu, select **Debug** -> **Start without Debugging**.

## Example Output
### Example Output for CPU on Linux
```
[100%] Built target hidden-markov-models
Device: Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz Intel(R) OpenCL
The Viterbi path is:
19 18 17 16 15 14 13 12 11 10
The sample completed successfully!
[100%] Built target run
```

## License
Code samples are licensed under the MIT license. See
[License.txt](License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](third-party-programs.txt).
