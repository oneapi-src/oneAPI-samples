# `Annotated Allocation` Sample
This tutorial demonstrates how to use the provided helper function to allocate host/shared memory with properties that match with the Avalon memory-mapped host interfaces on the IP components produced by the Intel® oneAPI DPC++/C++ Compiler.

| Optimized for                     | Description
---                                 |---
| What you will learn               | How to use the provided helper funtion to allocate host/shared memory with properties for your FPGA IP components
| Time to complete                  | 15 minutes
| Category             | Concepts and Functionality

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware             | Intel® Agilex® 7, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.

> **Note**: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

> **Note**: This tutorial will not work for a Full System compile as it demonstrates a SYCL HLS flow specific feature.

This sample is part of the FPGA code samples.
It is categorized as a Tier 2 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.


## Purpose
In the `hls_flow_interface/mmhost` sample, we demonstrated the usage of `annotated_arg` class in customizing Avalon memory-mapped interfaces for the FPGA IP component.To learn more about using `annotated_arg` class to customize Avalon memory-mapped interfaces, refer to the code sample [`hls_flow_interface/mmhost`](https://github.com/oneapi-src/oneAPI-samples/tree/development/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/hls_flow_interfaces/mmhost).

When testing the functionality and performance of the IP component using FPGA simulation, it is the responsibility of the SYCL host code to provide valid runtime values that matches with the properties specified in the `annotated_arg` type. This tutorial shows how to use the helper function `alloc_annotated` (defined in the utility header file "annotation_class_util.hpp") to ensure such requirement. 


### Using `allocated_alloc` to allocate host/shared memory with properties that match with Avalon memory-mapped host interfaces

In the example, the device code defines a SYCL kernel functor that computes the addition of two vectors as follows.

```c++
constexpr int kBL1 = 1;
constexpr int kBL2 = 2;
constexpr int kBL3 = 3;
constexpr int kAlignment = 4;

using arg_a_t= sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBL1>,
        sycl::ext::oneapi::experimental::alignment<kAlignment> })>;
using arg_b_t = sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBL2>,
        sycl::ext::oneapi::experimental::alignment<kAlignment>})>;
using arg_c_t = sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBL3>,
        sycl::ext::oneapi::experimental::alignment<kAlignment>})>;

struct VectorAdd {
  arg_a_t a;
  arg_b_t b;
  arg_c_t c;

  int size;

  void operator()() const {
    for (int i = 0; i < size; i++) {
      c[i] = a[i] + b[i];
    }
  }
};
```

The kernel arguments are all annotated_arg specified with properties `buffer location` and `alignment`. To test the IP component in the FPGA simulation, the SYCL host code is required to provide for the kernel arguments the runtime values that match with the `buffer location` and `alignment`. The helper function `alloc_annotated` (in `fpga_tools` namespace) provides this functionality with code simplicity, as it takes the annotated_arg type as the template parameter and return the an instance of the same annotated_arg type, as the following code snippet shows

```c++
#include "annotated_class_util.hpp"

constexpr int kN = 8;
arg_a_t array_a = fpga_tools::alloc_annotated<arg_a_t>(kN, q);
arg_b_t array_b = fpga_tools::alloc_annotated<arg_b_t>(kN, q);
arg_c_t array_c = fpga_tools::alloc_annotated<arg_c_t>(kN, q);

// Initialize the host array
...

q.single_task(VectorAdd{array_a, array_b, array_c, kN}).wait();
```

Note that "annotated_alloc" function only takes `annotated_arg` with pointer type as valid template parameters.


## Building the `allocated_allocation` Sample
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


### On Linux*
This design uses CMake to generate a build script for GNU/make.

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake .. 
   ```

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```

3. Compile the design using `make`.
   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      make fpga_sim
      ```
   3. Generate HTML performance report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      make report
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```

   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
3. Compile the design using `nmake`.
   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      nmake fpga_sim
      ```
   3. Generate HTML performance report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      nmake report
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
   > **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `allocated_allocation` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
   ```
   ./allocated_alloc.fpga_emu
   ```

2. Run the sample on the FPGA simulator device (the kernel executes in a simulator):
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./allocated_alloc.fpga_sim
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
   ```
   allocated_alloc.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device (the kernel executes in a simulator):
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   allocated_alloc.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

### Example of Output

```
Running on device: Intel(R) FPGA Emulation Device
Elements in vector : 8
PASSED
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
