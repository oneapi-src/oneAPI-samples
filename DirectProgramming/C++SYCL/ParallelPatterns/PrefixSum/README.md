# `Prefix Sum` Sample
This code sample demonstrates how to implement parallel prefix sum SYCL*-compliant code to
offload the computation to a GPU. In this implementation, a random sequence of 2**n elements is given (n is a positive number) as input. The algorithm computes the prefix sum in parallel. The result sequence is in ascending order.

| Property                | Description
|:---                     |:---
| What you will learn     | How to offload computations using the Intel® oneAPI DPC++/C++ Compiler
| Time to complete        | 15 minutes

## Purpose
Given a randomized sequence of numbers x0, x1, x2, ..., xn, this algorithm computes and returns a new sequence y0, y1, y2, ..., yn so that:

y0 = x0 <br>
y1 = x0 + x1 <br>
y2 = x0 + x1 + x2 <br>
..... <br>
yn = x0 + x1 + x2 + ... + xn

The following pseudo code shows how to compute prefix sum in parallel. **n** is power of 2 (1, 2, 4 , 8, 16, ...):

```cpp
for i from 0 to  [log2 n] - 1 do
   for j from 0 to (n-1) do in parallel
     if j<2^i then
       x_{j}^{i+1} <- x_{j}^{i}}
     else
       x_{j}^{i+1} <- x_{j}^{i} + x_{j-2^{i}}^{i}}

```
In the pseudo code shown above, the notation $x_{j}^{i}$ means the value of the jth element of array x in timestep i. Given n processors to perform each iteration of the inner loop in constant time, the algorithm runs in $O(log n)$ time, which is the number of iterations of the outer loop.

## Prerequisites
| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04 <br> Windows* 10
| Hardware                | Skylake with GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic concepts explained in the code include device selector, buffer, accessor, kernel, and command groups.

The code attempts to execute on an available GPU and the code falls back to the system CPU if a compatible GPU is not detected.

## Building the `PrefixSum` Program for CPU and GPU

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
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

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
3. From the top menu, select **Debug** > **Start without Debugging**.

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild PrefixSum.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Running the sample
### Application Parameters

The input values for `<exponent>` and `<seed>` are configurable. Default values for the sample are `<exponent>` = 21 and `<seed>` = 47.

Usage: `PrefixSum <exponent> <seed>`

- `<exponent>` is a positive number. (The length of the sequence is
2**exponent.)
- `<seed>` is the seed used by the random generator to generate the randomness.

The sample offloads the computation to the GPU and performs the verification
of the results in the CPU. The results are verified if yk = yk-1 + xk match. If the results are matched, and the ascending order is verified, and the application displays a “Success!” message.

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
2. Run the program with the default inputs.
```
PrefixSum.exe 21 47
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
