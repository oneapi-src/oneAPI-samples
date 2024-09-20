# `Merge SPMV` Sample
The `Merge SPMV` sample (also called the Sparse Matrix Vector sample) illustrates how to implement a merge-based sparse matrix and vector multiplication algorithm.

| Property               | Description
|:---                    |:---
| What you will learn    | How to offload compute intensive parts of an application using lambda kernel, and how to measure kernel execution times
| Time to complete       | 15 minutes

## Purpose
Sparse linear algebra algorithms are common in high-performance computing in fields like machine learning and computational science. This sample code implements a merge-based sparse matrix and vector multiplication algorithm. The input matrix is in compressed sparse row format. The sample code uses a parallel merge model to enable the application to offload compute intensive operation to the GPU efficiently.

## Prerequisites
| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04 <br> Windows* 10, 11
| Hardware                | Skylake with GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
This sample demonstrates the concepts of using a device selector, unified shared memory, kernel, and command groups in order to implement a solution using a parallel merge method.

To allow comparison between methods, the application runs sequentially and in parallel with run times for each method displayed in the output. The program output also includes the device on which code ran.

> **Note**: The workgroup size requirement is **256**. If your hardware cannot support this size, the application shows an error.

Compressed Sparse Row (CSR) representation for sparse matrix contains three components:
- Nonzero values
- Column indices
- Row offsets

You can think of both the row offsets and values indices as sorted arrays. In concept, the  progression of the computation is similar to merging two sorted arrays.

In the parallel merge, each thread independently identifies its scope of the merge and then performs only the amount of work that belongs to this thread.

The program will attempt to run on a compatible GPU. If a compatible GPU is not detected or available, the code will execute on the CPU instead.

## Build the `Merge SPMV` Program for CPU and GPU

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
4. From the top menu, select **Debug** > **Start without Debugging**.

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild merge-spmv.sln /t:Rebuild /p:Configuration="Release"`


#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Merge SPMV` Program
### On Linux
1. Run the program.
   ```
   make run
   ```
   Alternatively, you can run the program directly, `./spmv`.
2. Clean the project files. (Optional)
   ```
   make clean
   ```
### On Windows
1. Change to the output directory.

2. Run the program.
   ```
   merge-spmv.exe
   ```

## Example Output
The following output is for a GPU. CPU results are similar.
```
Device: Intel(R) Gen9
Compute units: 24
Work group size: 256
Repeating 16 times to measure run time ...
Iteration: 1
Iteration: 2
Iteration: 3
Iteration: 4
Iteration: 5
Iteration: 6
Iteration: 7
Iteration: 8
Iteration: 9
Iteration: 10
Iteration: 11
Iteration: 12
Iteration: 13
Iteration: 14
Iteration: 15
Iteration: 16
Successfully completed sparse matrix and vector multiplication!
Time sequential: 0.00436269 sec
Time parallel: 0.00909913 sec
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
