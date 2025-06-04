# `Bitonic Sort` Sample
This code sample demonstrates how to use a bitonic sort using SYCL* to offload the computation to a GPU. In this implementation, a random sequence of 2**n elements is provided as input (n is a positive number), and the algorithm sorts the sequence in parallel. The result sequence is ascending order.

| Property                | Description
|:---                     |:---
| What you will learn     | Implement bitonic sort for CPU and GPU
| Time to complete        | 15 minutes

## Purpose
The algorithm converts a randomized sequence of numbers into a bitonic sequence (two ordered sequences) and then merges these two ordered sequences into an ordered sequence.

The bitonic sort algorithm works according to the following summary:
- The algorithm decomposes the randomized sequence of size 2\*\*n into 2\*\*(n-1) pairs where each pair consists of 2 consecutive elements. Each pair is a bitonic sequence.

- **Step 0**: For each pair (sequence of size 2), the two elements are swapped so that the two consecutive pairs form a bitonic sequence in increasing order. The next two pairs form the second bitonic sequence in decreasing order. The next two pairs form the third bitonic sequence in increasing order, and so forth. At the end of this step, there are 2\*\*(n-1) bitonic sequences of size 2, and the sequences follow an order of increasing, decreasing, increasing, ..., decreasing. They form 2\*\*(n-2) bitonic sequences of size 4.

- **Step 1**: For each new 2\*\*(n-2) bitonic sequences of size 4, (each new sequence consists of 2 consecutive previous sequences), the algorithm swaps the elements so that at the end of step 1, there are 2\*\*(n-2) bitonic sequences of size 4. The sequences follow an order: increasing, decreasing, increasing, ..., decreasing. They form 2\*\*(n-3) bitonic sequences of size 8. The same logic applies until the algorithm reaches the last step.

- **Step n**: In the last step, there is one bitonic sequence of size 2\*\*n. The elements in the sequence are swapped until the sequences are in increasing order.

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic SYCL* implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups. Unified Shared Memory (USM) and Buffer Object are used for data management.

The code attempts to execute on an available GPU and it will fall back to the system CPU if it cannot detect a compatible GPU.

## Build the `Bitonic Sort` Program for CPU and GPU

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
3. Right-click on the project in **Solution Explorer** and select **Rebuild**.

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command:
     ```
     MSBuild bitonic-sort.sln /t:Rebuild /p:Configuration="Release"
     ```

### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics
Utility for Intel® oneAPI Toolkits, which provides system checks to find
missing dependencies and permissions errors. See [Diagnostics Utility for
Intel® oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


## Run the `Bitonic Sort` Sample

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
2. Run the executable with the default exponent and seed values.
   ```
   bitonic-sort 21 47
   ```

### Application Parameters
The input values for `<exponent>` and `<seed>` are configurable. Default values for the sample are `<exponent>` = 21 and `<seed>` = 47.

Usage: `bitonic-sort <exponent> <seed>`

where:

- `<exponent>` is a positive number. The according length of the sequence is
  2**exponent.
- `<seed>` is the seed used by the random generator to generate the randomness.


The sample offloads the computation to GPU and then performs the computation in
serial on the CPU, and then compares the results for the parallel and serial runs. If the results are matched and the ascending order is verified, the application will display a “Success!” message.

## Output Example
```
Array size: 2097152, seed: 47
Device: Intel(R) Gen9 HD Graphics NEO
Warm up ...
Kernel time using USM: 0.248422 sec
Kernel time using buffer allocation: 0.253364 sec
CPU serial time: 0.628803 sec

Success!
```

## License
Code samples are licensed under the MIT license. See
[License.txt](License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](third-party-programs.txt).
