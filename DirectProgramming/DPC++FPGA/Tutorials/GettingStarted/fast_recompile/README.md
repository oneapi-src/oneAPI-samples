
# Separating Host and Device Code Compilation
This FPGA tutorial demonstrates how to separate the compilation of a program's host code and device code to save development time.  It's recommended to read the 'fpga_compile' code sample before this one.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Why to separate host and device code compilation in your FPGA project <br> How to use the `-reuse-exe` and device link methods <br> Which method to choose for your project
| Time to complete                  | 15 minutes



## Purpose
Intel® oneAPI DPC++ Compiler only supports ahead-of-time (AoT) compilation for FPGA, which means that an FPGA device image is generated at compile time. The FPGA device image generation process can take hours to complete. Suppose you make a change that is exclusive to the host code. In that case, it is more efficient to recompile your host code only, re-using the existing FPGA device image and circumventing the time-consuming device compilation process.

The compiler provides two different mechanisms to separate device code and host code compilation.
* Passing the `-reuse-exe=<exe_name>` flag to `dpcpp` instructs the compiler to attempt to reuse the existing FPGA device image.
* The more explicit "device link" method requires you to separate the host and device code into separate files. When a code change only applies to host-only files, an FPGA device image is not regenerated.

This tutorial explains both mechanisms and the pros and cons of each. The included code sample demonstrates the device link method but does **not** demonstrate the use of the `-reuse-exe` flag.

### Using the `-reuse-exe` flag

If the device code and options affecting the device have not changed since the previous compilation, passing the `-reuse-exe=<exe_name>` flag to `dpcpp` instructs the compiler to extract the compiled FPGA binary from the existing executable and package it into the new executable, saving the device compilation time.

**Sample usage:**

```
# Initial compilation
dpcpp <files.cpp> -o out.fpga -Xshardware -fintelfpga
```
The initial compilation generates an FPGA device image, which takes several hours. Now, make some changes to the host code.
```
# Subsequent recompilation
dpcpp <files.cpp> -o out.fpga -reuse-exe=out.fpga -Xshardware -fintelfpga
```
If `out.fpga` does not exist, `-reuse-exe` is ignored and the FPGA device image is regenerated. This will always be the case the first time a project is compiled.

If `out.fpga` is found, the compiler checks whether any changes affecting the FPGA device code have been made since the last compilation. If no such changes are detected, the compiler reuses the existing FPGA binary, and only the host code is recompiled. The recompilation process takes a few minutes. Note that the device code is partially re-compiled (similar to a report flow compile) to check that the FPGA binary can safely be reused.

If `out.fpga` is found but the compiler cannot prove that the FPGA device code will yield a result identical to the last compilation, a warning is printed and the FPGA device code is fully recompiled. Since the compiler checks must be conservative, spurious recompilations can sometimes occur when using `-reuse-exe`.

### Using the device link method

The program accompanying this tutorial is separated into two files, `host.cpp` and `kernel.cpp`. Only the `kernel. cpp` file contains device code.

In the normal compilation process, FPGA device image generation happens at link time. As a result, any change to either `host.cpp` or `kernel.cpp` will trigger an FPGA device image's regeneration.

```
# normal compile command
dpcpp -fintelfpga host.cpp kernel.cpp -Xshardware -o link.fpga
```

The following graph depicts this compilation process:

![](normal_compile.png)


If you want to iterate on the host code and avoid long compile time for your FPGA device, consider using a device link to separate device and host compilation:

```
# device link command
dpcpp -fintelfpga -fsycl-link=image <input files> [options]
```

The compilation is a 3-step process:

1. Compile the device code:

   ```
   dpcpp -fintelfpga -fsycl-link=image kernel.cpp -o dev_image.a -Xshardware
   ```
   Input files should include all source files that contain device code. This step may take several hours.


2. Compile the host code:

   ```
   dpcpp -fintelfpga host.cpp -c -o host.o
   ```
   Input files should include all source files that only contain host code. This takes seconds.


3. Create the device link:

   ```
   dpcpp -fintelfpga host.o dev_image.a -o fast_recompile.fpga
   ```
   The input should have N (N >= 0) host object files *(.o)* and one device image file *(.a)*. This takes seconds.

**NOTE:** You only need to perform steps 2 and 3 when modifying host-only files.

The following graph depicts the device link compilation process:

![](device_link.png)

### Which method to use?
Of the two methods described, `-reuse-exe` is easier to use. It also allows you to keep your host and device code as single source, which is preferred for small programs.

For larger and more complex projects, the device link method has the advantage of giving you complete control over the compiler's behavior.
* When using `-reuse-exe`, the compiler must partially recompile and then analyze the device code to ensure that it is unchanged. This takes several minutes for larger designs. Compiling separate files does not incur this extra time.
* When using `-reuse-exe`, you may occasionally encounter a "false positive" where the compiler wrongly believes that it must recompile your device code. In a single source file, the device and host code are coupled, so some changes to the host code _can_ change the compiler's view of the device code. The compiler will always behave conservatively and trigger a full recompilation if it cannot prove that reusing the previous FPGA binary is safe. Compiling separate files eliminates this possibility.

### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.

## Key Concepts
* Why to separate host and device code compilation in your FPGA project
* How to use the `-reuse-exe` and device link methods
* Which method to choose for your project

## Building the `fast_recompile` Tutorial
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
>For more information on environment variables, see **Use the setvars Script** for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.


### Using Visual Studio Code*  (Optional)

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


After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

     **NOTE:** For the FPGA emulator target and the FPGA target, the device link method is used.
2. Compile the design through the generated `Makefile`. The following build targets are provided:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/fast_recompile.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

> **Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Troubleshooting
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./fast_recompile.fpga_emu     (Linux)
     fast_recompile.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./fast_recompile.fpga         (Linux)
     fast_recompile.fpga.exe       (Windows)
     ```

### Example of Output
```
PASSED: results are correct
```
### Discussion of Results
Try modifying `host.cpp` to produce a different output message. Then, perform a host-only recompile via the device link method to see how quickly the design is recompiled.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).