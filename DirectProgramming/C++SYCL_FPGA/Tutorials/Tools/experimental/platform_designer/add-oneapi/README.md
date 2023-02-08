# `Platform Designer` Sample - oneAPI code

This oneAPI project will be compiled and deployed in an Intel® Platform Designer system in the Intel Quartus® Prime software.

## Building the `add-oneapi` Tutorial

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. 
> Set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation every time you open a new terminal window. 
> This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

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

   | Target          | Expected Time  | Output                                                                       | Description
   |:---             |:---            |:---                                                                          |:---
   | `make fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `make report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.
   | `make fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `make fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime. If you specified a BSP with `FPGA_DEVICE`, this will generate an FPGA image that you can run on the corresponding accelerator board.
   | `make fpga_ip_export` | Minutes  | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL that may be exported to Intel® Quartus Prime software

   The `fpga_emu`, `fpga_sim` and `fpga` targets produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu`, `TARGET_NAME.fpga_sim`, and `TARGET_NAME.fpga`, where `TARGET_NAME` is the value you specify in `CMakeLists.txt`.

### On a Windows* System
This design uses CMake to generate a build script for  `nmake`.

1. Generate the `Makefile` by running `cmake`.

   ```bash
   mkdir build
   cd build
   ```

   To compile for the Intel® Arria 10 SoC Development Kit, run `cmake` using the command:

   ```bash
   cmake -G "NMake Makefiles" ..
   ```

   You can also compile for a different FPGA device. Run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<other FPGA (e.g. Agilex), or a part number>
   ```

   > **NOTE**: This design will **not** work on an Intel® acceleration card, because this design is for custom IP Authoring only.


2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target           | Expected Time  | Output                                                                       | Description
   |:---              |:---            |:---                                                                          |:---
   | `nmake fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `nmake report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package.
   | `nmake fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `nmake fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime. If you specified a BSP with `FPGA_DEVICE`, this will generate an FPGA image that you can run on the corresponding accelerator board.
   | `nmake fpga_ip_export` | Minutes  | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL that may be exported to Intel® Quartus Prime software

   The `fpga_emu`, `fpga_sim`, and `fpga` targets also produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu.exe`, `TARGET_NAME.fpga_sim.exe`, and `TARGET_NAME.fpga.exe`, where `TARGET_NAME` is the value you specify in `src/CMakeLists.txt`.

   > **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `add-oneapi` Executable

### On Linux
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./add.fpga_emu
   ```

2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./add.fpga_sim
   ```

### On Windows
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   add.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   add.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).