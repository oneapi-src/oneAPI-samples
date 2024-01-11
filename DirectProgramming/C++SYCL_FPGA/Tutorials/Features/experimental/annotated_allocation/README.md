# `Annotated Allocation` Sample

This tutorial demonstrates how to use the provided helper function to allocate host/shared memory with properties that match with the Avalon memory-mapped host interfaces on the IP components produced by the Intel® oneAPI DPC++/C++ Compiler.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware                          | Intel® Agilex® 7, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | How to use the provided helper funtion to allocate host/shared memory with properties for your FPGA IP components
| Time to complete                  | 10 minutes



> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> To use the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

> **Note**: This tutorial will not work for a Full System compile as it demonstrates a SYCL HLS flow specific feature.

## Prerequisites
This sample is part of the FPGA code samples.
It is categorized as a Tier 2 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.


## Purpose
In the `hls_flow_interface/mmhost` sample, we demonstrated the usage of `annotated_arg` class in customizing Avalon memory-mapped interfaces for the FPGA IP component. To learn more about using `annotated_arg` class to customize Avalon memory-mapped interfaces, refer to the code sample [hls_flow_interface/mmhost](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/hls_flow_interfaces/mmhost).

When testing the functionality and performance of the IP component using FPGA simulation, it is the responsibility of the SYCL host code to provide valid runtime values that matches with the properties specified in the `annotated_arg` type. This tutorial shows how to use the helper function `alloc_annotated` (defined in the utility header file "annotation_class_util.hpp") to ensure such requirement. 


### Using `annotated_alloc` to allocate host/shared memory with properties that match with Avalon memory-mapped host interfaces

In the example, the device code defines a SYCL kernel functor that computes the addition of two vectors as follows.

```c++
using annotated_arg_t= sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<0>,
        sycl::ext::oneapi::experimental::alignment<256> })>;

struct  {
  annotated_arg_t a;

  int size;

  void operator()() const {
   #pragma unroll 8
    for (int i = 0; i < size; i++) {
      a[i] *= 2;
    }
  }
};
```

The kernel arguments are all annotated_arg specified with properties `buffer location` and `alignment`. To test the IP component in the FPGA simulation, the SYCL host code is required to provide for the kernel arguments the runtime values that match with the `buffer location` and `alignment`. The helper function `alloc_annotated` (in `fpga_tools` namespace) provides this functionality with code simplicity, as it takes the annotated_arg type as the template parameter and return the an instance of the same annotated_arg type, as the following code snippet shows

```c++
#include "annotated_class_util.hpp"

constexpr int kN = 8;
annotated_arg_t array_a = fpga_tools::alloc_annotated<arg_a_t>(kN, q);

// Initialize the host array
...

q.single_task(VectorAdd{array_a, array_b, array_c, kN}).wait();
```

> **Note**: "annotated_alloc" function only takes `annotated_arg` with pointer type as valid template parameters.

## Building the `annotated_allocation` Sample
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

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake .. 
   ```

   You can create a debuggable binary by setting `CMAKE_BUILD_TYPE` to `Debug`:
   ```
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   ```
   If you want to use the `report`, `fpga_sim`, or `fpga` flows, you should switch the `CMAKE_BUILD_TYPE` back to `Release`:
   ```
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```

3. Compile the design with the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target          | Expected Time  | Output                                                                       | Description
   |:---             |:---            |:---                                                                          |:---
   | `make fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `make report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package. The generated RTL may be exported to Intel® Quartus Prime software.
   | `make fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `make fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime.

   The `fpga_emu` and `fpga_sim` targets produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu`, `TARGET_NAME.fpga_sim`, and `TARGET_NAME.fpga`, where `TARGET_NAME` is the value you specify in `CMakeLists.txt`.
   You can see a listing of the commands that are run:
   ```bash
   build $> make report
   [ 33%] To compile manually:
   /[ ... ]/linux64/bin/icpx -I../../../../include -fsycl -fintelfpga    -Wall -qactypes -DFPGA_HARDWARE -c ../src/annotated_alloc.cpp -o CMakeFiles/report.dir/src/ annotated_alloc.cpp.o

   To link manually:
   /[ ... ]/linux64/bin/icpx -fsycl -fintelfpga -Xshardware  -Xstarget=Agilex7 -fsycl-link=early -o annotated_alloc.report CMakeFiles/report.dir/src/  annotated_alloc.cpp.o
   [ ... ]

   [100%] Built target report
      ```

### On a Windows* System
This design uses CMake to generate a build script for  `nmake`.

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```

   You can create a debuggable binary by setting `CMAKE_BUILD_TYPE` to `Debug`:
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" .. -DCMAKE_BUILD_TYPE=Debug
   ```
   If you want to use the `report`, `fpga_sim`, or `fpga` flows, you should switch the `CMAKE_BUILD_TYPE` back to `Release``:
   ```
   cmake -G "NMake Makefiles" .. -DCMAKE_BUILD_TYPE=Release
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```

3. Compile the design with the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Target           | Expected Time  | Output                                                                       | Description
   |:---              |:---            |:---                                                                          |:---
   | `nmake fpga_emu` | Seconds        | x86-64 binary                                                                | Compiles the FPGA device code to the CPU. Use the Intel® FPGA Emulation Platform for OpenCL™ software to verify your SYCL code’s functional correctness.
   | `nmake report`   | Minutes        | RTL + FPGA reports                                                           | Compiles the FPGA device code to RTL and generates an optimization report that describes the structures generated on the FPGA, identifies performance bottlenecks, and estimates resource utilization. This report will include the interfaces defined in your selected Board Support Package. The generated RTL may be exported to Intel® Quartus Prime software.
   | `nmake fpga_sim` | Minutes        | RTL + FPGA reports + x86-64 binary                                           | Compiles the FPGA device code to RTL and generates a simulation testbench. Use the Questa*-Intel® FPGA Edition simulator to verify your design.
   | `nmake fpga`     | Multiple Hours | Quartus Place & Route (Full accelerator) + FPGA reports + x86-64 host binary | Compiles the FPGA device code to RTL and compiles the generated RTL using Intel® Quartus® Prime.

   The `fpga_emu` and `fpga_sim` targets also produce binaries that you can run. The executables will be called `TARGET_NAME.fpga_emu.exe`, `TARGET_NAME.fpga_sim.exe`, and `TARGET_NAME.fpga.exe`, where `TARGET_NAME` is the value you specify in `CMakeLists.txt`.
   > **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.
   You can see a listing of the commands that are run:
   ```bash
   build> nmake report

   [ 33%] To compile manually:
   C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/bin/icx-cl.exe -I../../../../include   -fsycl -fintelfpga -Wall /EHsc -Qactypes -DFPGA_HARDWARE -c ../src/annotated_alloc.cpp -o  CMakeFiles/report.dir/src/annotated_alloc.cpp.obj

   To link manually:
   C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/bin/icx-cl.exe -fsycl -fintelfpga   -Xshardware -Xstarget=Agilex7 -fsycl-link=early -o annotated_alloc.report.exe   CMakeFiles/report.dir/src/annotated_alloc.cpp.obj
   [ ... ]
   [100%] Built target report
      ```

## Run the `annotated_allocation` Executable

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
   ```
   ./annotated_alloc.fpga_emu
   ```

2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./annotated_alloc.fpga_sim
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
   ```
   annotated_alloc.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   annotated_alloc.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

## Example Output

```
Running on device: Intel(R) FPGA Emulation Device
Elements in vector : 8
Elements in vector : 8
PASSED
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
