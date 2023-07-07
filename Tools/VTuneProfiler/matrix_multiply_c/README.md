# `Matrix Multiply` Sample
The sample included with the Intel VTune Profiler installer, containing multiple implementations of matrix multiplication. This sample code is implemented in the C language for CPU.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br> Windows* 10
| Hardware                          | Intel CPU
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br> Intel&reg; VTune&trade; Profiler
| What you will learn               | How to profile an application using Intel&reg; VTune&trade; Profiler
| Time to complete                  | 15 minutes

## Purpose

The Matrix Multiplication sample performs basic matrix multiplication. Six
versions are provided with various levels of CPU optimizations.

## Key Implementation details

By default a square matrix is used with size of 2048. You may want to increase matrix size by redefining the macro MAXTHREADS in the multiply.h. However, make sure the size is multiple of # of threads. It's made for simplicity.

By default # of threads executed is equal to number of CPU cores available in the system (defined in run-time). You may want to limit max number of created threads by modifying the MAXTHREADS. 

Default threading model is native, i.e. pthreads on Linux and Win32 threads on Windows.

OpenMP threading model is available.

For the MKL-based kernel consider either MKL's multithreaded or singlethreaded implementation. Make the kernel multithreaded by yourself in the latter case. 

In order to change the threading model:

In Windows project: select either of Release (default), Release_OMP, or Release_MKL configuration.

In Linux Makefile: select PARAMODEL either of USE_THR, USE_OMP, or USE_MKL.

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## How to Build

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
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

This sample contains six versions of matrix multiplication:

- `multiply0` – Basic serial implementation
- `multiply1` – Basic multithreaded implementation
- `multiply2` – Optimized implementation with Loop interchange and vectorization (use Compiler vectorization options)
- `multiply3` – Optimized implementation with adding Cache blocking and data alignment (add ALIGNED macro to Compiler preprocessor)
- `multiply4` – Optimized implementation with matrix transposition and loop unrolling
- `multiply5` – Most optimal version with using Intel MKL (link the MKL library to the project and add the USE_MKL macro to Compiler preprocessor)
  
Note: multiply kernels availability depends on threading model you choose.

Edit the line in `multiply.h` to select the version of the multiply function:
`#define MULTIPLY multiply1`.

### On a Linux* System
	To build with the Intel Compiler:
	cd <sample dir>\linux
	make icx

    Clean the program
    make clean

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### On a Windows* System Using Visual Studio 2017 or newer
   * Open Visual Studio 2017 or later
     * Select Menu "File > Open > Project/Solution", find "matrix\vc16" folder and select "matrix.sln"
     * Select Menu "Project > Build" to build the selected configuration
     * Select Menu "Debug > Start Without Debugging" to run the program

### on Windows - command line - Build the program using MSBuild
- DPCPP Configurations:
  - Release - `MSBuild matrix.sln /t:Rebuild /p:Configuration="Release"`
  - Debug - `MSBuild matrix.sln /t:Rebuild /p:Configuration="Debug"`


## Example of Output
```
./matrix

Addr of buf1 = 0x7f46bafff010
Offs of buf1 = 0x7f46bafff180
Addr of buf2 = 0x7f46b8ffe010
Offs of buf2 = 0x7f46b8ffe1c0
Addr of buf3 = 0x7f46b6ffd010
Offs of buf3 = 0x7f46b6ffd100
Addr of buf4 = 0x7f46b4ffc010
Offs of buf4 = 0x7f46b4ffc140
Threads #: 16 Pthreads
Matrix size: 2048
Using multiply kernel: multiply1
Freq = 2.847324 GHz
Execution time = 0.769 seconds
MFLOPS: 22331.479 mflops
```

## Running an Intel&reg; VTune&trade; Profiler analysis
```
vtune -collect hotspots -- ./matrix
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).


 
