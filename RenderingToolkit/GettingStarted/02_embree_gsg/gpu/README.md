# `Getting Started` GPU Sample for Intel&reg; oneAPI Rendering Toolkit (Render Kit): Intel&reg; Embree

This sample program, `minimal_sycl`, performs two ray-to-triangle-intersect tests with the Intel&reg; Embree API. One test is a successful intersection, and the second test misses. The program performs intersection tests on the GPU device. The program writes output results to the console (stdout).

| Area                 | Description
|:---                  |:---
| What you will learn  | How to build and run a basic sample program for Intel&reg; GPUs using the Intel Embree API from the Render Kit.
| Time to complete     | 5 minutes

## Prerequisites

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Ubuntu* 22.04 <br> RHEL 8.5, 8.6 (or compatible) <br>Windows* 10 64-bit 20H2 or higher<br>Windows 11* 64-bit
| Hardware                          | Intel&reg; Arc&trade; GPU or higher, compatible with Intel Xe-HPG architecture
| Compiler Toolchain                | **Windows\***: Intel&reg; oneAPI DPC++ Compiler 2023.0 or higher, MSVS 2019 or higher with Windows* SDK and CMake*<br>**Linux\***: Intel&reg; oneAPI DPC++ Compiler 2023.0 or higher, C++17 system compiler (for example g++), and CMake*
| Libraries                         | Intel&reg; oneAPI DPC++ Compiler and Runtime Library (Base Toolkit)<br>Intel&reg; oneAPI Rendering Toolkit (Render Kit), includes Embree
| GPU Configuration                 | **System BIOS**: [Quick Start](https://www.intel.com/content/www/us/en/support/articles/000091128/graphics.html) <br> **Windows\***: [Drivers for Intel&reg; Graphics products](https://www.intel.com/content/www/us/en/support/articles/000090440/graphics.html ) <br> **Linux\***: [Install Guide](https://dgpu-docs.intel.com/installation-guides/index.html#) 
## Key Implementation Details

- Embree computation is offloaded to the GPU device via the SYCL* implementation.
- The SYCL* implementation is provided with Intel DPC/C++ runtime libraries. It uses the system graphics driver.
- Note: Embree releases with Beta GPU functionality may report a Beta message.

## Build and Run the `Getting Started` Sample Program

### Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
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
2. Change to the GPU sample program directory:
```
cd <path-to-oneAPI-samples>\RenderingToolkit\GettingStarted\02_embree_gsg
cd gpu
```

3. Build and run the `Getting Started` program:
```
mkdir build
cd build
cmake -G"Visual Studio 17 2022" -A x64 -T"Intel(R) oneAPI DPC++ Compiler 2023" ..
cmake --build . --config Release
cmake --install . --config Release
cd ..\bin
.\minimal_sycl.exe
```

> Note: Replace the generator flag *"Visual Studio 17 2022"* with *"Visual Studio 16 2019"* as needed for MSVS 2019.

4. Review the terminal output (stdout).

### On Linux*

1. Start a new Terminal session. Ensure environment variables are set.https://www.intel.com/content/www/us/en/support/articles/000091128/graphics.html
2. Change to the GPU sample program directory.
```
cd <path-to-oneAPI-samples>/RenderingToolkit/GettingStarted/02_embree_gsg
cd gpu
```
3. Build and run the `Getting Started` program:
```
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
cmake --install .
cd ../bin
./minimal_sycl
```
4. Review the terminal output (stdout).

### Example Output
```
0.330000, 0.330000, -1.000000: Found intersection on geometry 0, primitive 0 at tfar=1.000000
1.000000, 1.000000, -1.000000: Did not find any intersection.

```

## Next Steps

Continue with the [Getting Started Guide](../../../GettingStarted/03_openvkl_gsg) or review the detailed [walkthrough tutorial programs](../../../Tutorial) using Embree to raytrace and pathtrace images.

## License

This code sample is licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
