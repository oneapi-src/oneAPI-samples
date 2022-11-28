# Platform Designer - oneAPI code

This oneAPI project will be compiled and deployed in an Intel® Platform Designer system in the Intel Quartus® Prime software.

## Building the `add-oneapi` Design

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `/opt/intel/oneapi/setvars.sh`
> - For private installations: `~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
>For more information on environment variables, see **Use the setvars Script** for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Running Samples in Intel® DevCloud
If running a sample in the Intel® DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).
When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.
### Using Visual Studio Code*  (Optional)
You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.
The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.
To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

Use these commands to run the design, depending on your OS.

### On a Linux* System 
This design uses CMake to generate a build script for GNU/make.

1. Generate the `Makefile` by running `cmake`.

   ```bash
   mkdir build
   cd build
   ```

   To compile for the Intel® Arria 10 SoC Development Kit, run `cmake` using the command:

   ```bash
   cmake ..
   ```

   You can also compile for a different FPGA device. Run `cmake` using the command:

   ```
   cmake .. -DFPGA_DEVICE=<other FPGA (e.g. Agilex), or a part number>
   ```

   > **NOTE**: This design will **not** work on an Intel® acceleration card, because this design is for custom IP Authoring only.

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target            | Expected Time  | Output                                                                       | Description                                                                                                                                                                                                                                                                                             |
   |-------------------|----------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   | `make fpga_emu`   | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.                                                                                                                                                |
   | `make report`     | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.          |
   | `make fpga_sim`   | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.                                                                                                                                                         |
   | `make fpga_ip_export` | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL that may be exported to Intel® Quartus Prime software                                                                                                                                                                                                              |
   | `make fpga`       | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and generate an FPGA image that you can run on a supported accelerator board.                                                                                                                                                                                      |

   The `fpga_emu`, `fpga_sim` and `fpga` targets produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu`, `TARGET_NAME.fpga_sim`, and `TARGET_NAME.fpga`, where `TARGET_NAME` is the value you specify in `src/CMakeLists.txt`.

### On a Windows* System
This design uses CMake to generate a build script for  `nmake`.

1. Generate the `Makefile` by running `cmake`.

   ```bash
   mkdir build
   cd build
   ```

   To compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```bash
   cmake -G "NMake Makefiles" ..
   ```

   You can also compile for a different FPGA device. Run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<other FPGA (e.g. Agilex), or a part number>
   ```

   > **NOTE**: This design will **not** work on an Intel® acceleration card, because this design is for custom IP Authoring only.


2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target                 | Expected Time  | Output                                                                       | Description                                                                                                                                                                                                                                                                                             |
   |------------------------|----------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   | `nmake fpga_emu`       | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.                                                                                                                                                |
   | `nmake report`         | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.          |
   | `nmake fpga_sim`       | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.                                                                                                                                                         |
   | `nmake fpga_ip_export` | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL that may be exported to Intel® Quartus Prime software                                                                                                                                                                                                              |
   | `nmake fpga`           | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and generate an FPGA image that you can run on a supported accelerator board.                                                                                                                                                                                      |

   The `fpga_emu`, `fpga_sim`, and `fpga` targets also produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu.exe`, `TARGET_NAME.fpga_sim.exe`, and `TARGET_NAME.fpga.exe`, where `TARGET_NAME` is the value you specify in `src/CMakeLists.txt`.

   > **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Additional Documentation
- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

### Troubleshooting
If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument (on Windows use `nmake` instead):

```
make VERBOSE=1
```

```
nmake VERBOSE=1
```

For more comprehensive troubleshooting, use the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors. [Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html).

## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):

   ```
   ./add.fpga_emu     (Linux)
   add.fpga_emu.exe   (Windows)
   ```

2. Run the sample on the FPGA simulator device:
   ```
   ./add.fpga_sim     (Linux)
   add.fpga_sim.exe   (Windows)
   ```

3. Run the sample on the FPGA device:
   ```
   ./add.fpga         (Linux)
   add.fpga.exe       (Windows)
   ```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).