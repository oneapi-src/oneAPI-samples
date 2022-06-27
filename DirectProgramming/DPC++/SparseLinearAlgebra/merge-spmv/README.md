# `Sparse Matrix Vector` Sample
Sparse Matrix Vector sample provides a SYCL*-compliant parallel implementation of a merge based sparse matrix and vector multiplication algorithm.

For comprehensive information in using oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide), and use search or the table of contents to find relevant information.

| Property                     | Description
|:---                               |:---
| What you will learn               | The Sparse Matrix Vector sample demonstrates the following using the Intel&reg; oneAPI DPC++/C++ Compiler <br>Offloading compute intensive parts of the application using lambda kernel <br>Measuring kernel execution time
| Time to complete                  | 15 minutes


## Purpose
Sparse linear algebra algorithms are common in high-performance computing, in fields as machine learning and computational science. In this sample, a merge based sparse matrix and vector multiplication algorithm is implemented. The input matrix is in compressed sparse row format. Use a parallel merge model enables the application to offload compute intensive operation to the GPU efficiently. For comparison, the application is run sequentially and in parallel with run times for each displayed in the application output. The device where the code is run is also identified.

The workgroup size requirement is **256**. If your hardware cannot support this, the application will present an error.

Compressed Sparse Row (CSR) representation for sparse matrix have three components:
- Nonzero values
- Column indices
- Row offsets

Both row offsets and values indices can be thought of as sorted arrays. The progression of the computation is similar to that of merging two sorted arrays at a conceptual level.

In parallel implementation, each thread independently identifies its scope of the merge and then performs only the amount of work that belongs to this thread in the cohort of threads.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br>Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler


## Key Implementation Details
Includes device selector, unified shared memory, kernel, and command groups in order to implement a solution using a parallel merge method in which each thread independently identifies its scope of the merge and then performs only the amount of work that belongs to this thread.

## Build the Program for CPU and GPU
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Use Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*

Perform the following steps:

1.  Build the program using the following `cmake` commands.
    ```
    $ cd merge-spmv
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```

2.  Run the program
    ```
    $ make run
    ```

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### On Windows* Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild merge-spmv.sln /t:Rebuild /p:Configuration="Release"`

### Run Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

## Run the Sample
### Example Output
```
Device: Intel(R) Gen9
Compute units: 24
Work group size: 256
Repeating 16 times to measure run time ...
Iteration: 1
Iteration: 2
Iteration: 3
...
Iteration: 16
Successfully completed sparse matrix and vector multiplication!
Time sequential: 0.00436269 sec
Time parallel: 0.00909913 sec
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).