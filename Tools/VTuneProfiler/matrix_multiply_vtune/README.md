# `Matrix Multiply` Sample
A sample containing multiple implementations of matrix multiplication. This sample code is implemented using SYCL*-compliant code for CPU and GPU.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br> Windows* 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br> Intel&reg; VTune&trade; Profiler
| What you will learn               | How to profile an application using Intel&reg; VTune&trade; Profiler
| Time to complete                  | 15 minutes

## Purpose

The Matrix Multiplication sample performs basic matrix multiplication. Three
versions are provided that use different SYCL features.

## Key Implementation details

The basic SYCL implementation explained in the code includes device selector,
buffer, accessor, kernel, and command groups. 

## Include Files 
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development
system.

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
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

This sample contains three versions of matrix multiplication:

- `multiply1` – basic implementation of matrix multiply using SYCL
- `multiply1_1` – basic implementation that replaces the buffer store with a local accessor “acc” to reduce memory traffic
- `multiply1_2` – basic implementation plus the local accessor and matrix tiling

Edit the line in `multiply.h` to select the version of the multiply function:
`#define MULTIPLY multiply1`.

### On a Linux* System
	To build SYCL version:
	cd <sample dir>
	cmake .
	make

    Clean the program
    make clean

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### On a Windows* System Using Visual Studio* Version 2019 or Newer

#### Command Line using MSBuild

1. DPCPP Configurations:
  - Release 
  ```
    MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Release"
  ```
  - Debug
  ```
    MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Debug"
  ```
2. Navigate to the Configuration folder (example: x64 folder)

3. Run the program: 
```
  matrix_multiply.exe
```

#### Visual Studio IDE

1. Open Visual Studio 2019 or later

2. Select Menu "File > Open > Project/Solution", find "matrix_multiply" folder and select "matrix_multiply.sln"

3. Select Menu "Build > Build Solution" to build the selected configuration

4. After Build, select Menu "Debug > Start Without Debugging" to run the program


## Example of Output
```
Address of buf1 = 00000252964F2040
Offset of buf1 = 00000252964F2180
Address of buf2 = 0000025296D09040
Offset of buf2 = 0000025296D091C0
Address of buf3 = 000002529751B040
Offset of buf3 = 000002529751B100
Address of buf4 = 0000025297D20040
Offset of buf4 = 0000025297D20140
Using multiply kernel: multiply1

Running on Intel(R) Iris(R) Xe Graphics

Elapsed Time: 0.42209s
```

## Running an Intel&reg; VTune&trade; Profiler analysis
```
vtune -collect gpu-hotspots -- ./matrix.dpcpp
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).