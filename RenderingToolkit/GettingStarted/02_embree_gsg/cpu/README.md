# `Getting Started` CPU Sample for Intel&reg; oneAPI Rendering Toolkit (Render Kit): Intel&reg; Embree

This sample program, `minimal`, performs two ray-to-triangle-intersect tests with the Intel Embree API. One test is a successful intersection, and the second test misses. The program performs intersection tests on the CPU device. The program writes output results to the console (stdout).

| Area                 | Description
|:---                  |:---
| What you will learn  | How to build and run a basic sample program for CPU using the Intel Embree API from the Render Kit.
| Time to complete     | 5 minutes

## Prerequisites

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br>CentOS* 8 (or compatible) <br>Windows* 10 or higher<br>macOS* 10.15+
| Hardware                          | Intel 64 Penryn or newer with SSE4.1 extensions; ARM64 with NEON extensions <br>(Optimized requirements: Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions)
| Compiler Toolchain                | **Windows\***: MSVS 2019 or higher with Windows* SDK and CMake*; Other platforms: C++11 compiler, a C99 compiler (for example gcc/c++/clang), and CMake*
| Libraries                         | Install Intel&reg; oneAPI Rendering Toolkit (Render Kit), includes Embree

## Build and Run the `Getting Started` Sample Program

### Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux* and MacOS*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

### On Windows*

1. Open a new **x64 Native Tools Command Prompt for MSVS 2022** (or MSVS 2019). Ensure environment variables are set.
2. Change to CPU sample program directory:
```
cd <path-to-oneAPI-samples>\RenderingToolkit\GettingStarted\02_embree_gsg
cd cpu
```
3. Build and run the `Getting Started` program:
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cmake --install . --config Release
cd ..\bin
.\minimal.exe
```

4. Review the terminal output (stdout).

```
0.000000, 0.000000, -1.000000: Found intersection on geometry 0, primitive 0 at tfar=1.000000
1.000000, 1.000000, -1.000000: Did not find any intersection.
```

### On Linux* and macOS*

1. Start a new Terminal session.  Ensure environment variables are set.
2. Change to CPU sample program directory:
```
cd <path-to-oneAPI-samples>/RenderingToolkit/02_embree_gsg
cd cpu
```
3. Build and run the `Getting Started` program:
```
mkdir build
cd build
cmake ..
cmake --build .
cmake --install .
cd ../bin
./minimal
```
4. Review the terminal output (stdout).

### Example Output
```
0.000000, 0.000000, -1.000000: Found intersection on geometry 0, primitive 0 at tfar=1.000000
1.000000, 1.000000, -1.000000: Did not find any intersection.

```

## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
