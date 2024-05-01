# `Annotated Classes Clean Coding` Sample

This Intel® FPGA tutorial demonstrates how to use the included `annotated_class_util.hpp` header file to write cleaner code. You can use this header file to write cleaner code as follows:
* simplify the declaration of annotated types with groups of properties by using the `properties_t` alias.
* create annotated pointers more neatly than the standard USM allocation APIs by using the `alloc_annotated` helper function.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware                          | Intel® Agilex® 7, Agilex® 5, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | How to use the provided helper funtion to allocate host/shared memory with properties for your FPGA IP components
| Time to complete                  | 10 minutes


> **Note**: Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> To use the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

> :warning: Make sure the GCC (for Linux) or Visual Studio (for Windows) in your system supports C++20 features. This requires GCC 9 or later (for Linux) or Visual Studio 2019 version 16.11 or later (for Windows).

> **Note**: This tutorial does not apply to the FPGA acceleration flow as it demonstrates a SYCL HLS flow specific feature.

## Prerequisites
This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstrates an advanced code optimization.

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.


## Purpose
The `annotated_arg` class can be used to customize Avalon memory-mapped interfaces for FPGA IP components. (To learn more about the `annotated_arg` class, refer to the code sample [mmhost](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/hls_flow_interfaces/mmhost)). This tutorial will further demonstrate how to use the helper code in `annotated_class_util.hpp` to clean up your code that uses `annotated_arg`. You may add `annotated_class_util.hpp` to your designs.


### Simplify the Declaration of Annotated Types with Groups of Properties by Using the `properties_t` Alias

Most oneAPI code assigns properties to an `annotated_arg` type using `decltype`. For example,
```c++
using ArgProps = decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::buffer_location<1>,
      sycl::ext::intel::experimental::awidth<32>,
      sycl::ext::intel::experimental::dwidth<256>,
      sycl::ext::oneapi::experimental::alignment<32>,
      sycl::ext::intel::experimental::maxburst<8>,
      sycl::ext::intel::experimental::latency<0>});
sycl::ext::oneapi::experimental::annotated_arg<int *, ArgProps> arg_x;
```

The header file "annotated_class_util.hpp" provides a type alias `properties_t` (in `fpga_tools` namespace) that lets you simplify your code as follows:
```c++
#include "annotated_class_util.hpp"

using ArgProps = fpga_tools::properties_t<
      sycl::ext::intel::experimental::buffer_location<1>,
      sycl::ext::intel::experimental::awidth<32>,
      sycl::ext::intel::experimental::dwidth<256>,
      sycl::ext::oneapi::experimental::alignment<32>,
      sycl::ext::intel::experimental::maxburst<8>,
      sycl::ext::intel::experimental::latency<0>>>;
sycl::ext::oneapi::experimental::annotated_arg<int *, ArgProps> arg_x;
```

> **Note**: to use the `properties_t` alias, you need to add the compiler flag `-std=c++20` to your compilation command. `-std=c++20` has been added to the `USER_FLAGS` variable in the CMakeLists.txt file of this tutorial:
> ```
> set(USER_FLAGS ${USER_FLAGS} -std=c++20)
>```

> **Note**: The `properties_t` alias can be used on other annotated classes that specify properties in form of `decltype(sycl::ext::oneapi::experimental::properties{...})`, such as [device_global](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/experimental/device_global) and [pipe](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/hls_flow_interfaces/streaming_data_interfaces), assuming you have included the header file "annotation_class_util.hpp" that appears in this code sample.


### Allocate Host/Shared Memory with Properties That Match with Avalon Memory-Mapped Host Interfaces by Using the Helper Function `alloc_annotated`

While SYCL device code must specify `annotated_arg` properties at **compile** time, the SYCL specification allows host code to specify these properties at **run** time. If you are designing and testing an IP component, this additional flexibility is of no use to you, so you can use the helper function `alloc_annotated()` (defined in the utility header file "annotation_class_util.hpp") to simplify your code, while still ensuring that the properties in your host allocations match the properties you defined in your device code.

In the example, the device code defines a SYCL kernel functor that multiplies a factor of 2 to an input vector as follows:

```c++
using annotated_arg_t= sycl::ext::oneapi::experimental::annotated_arg<
      int *, fpga_tools::properties_t<
            sycl::ext::intel::experimental::buffer_location<0>,
            sycl::ext::oneapi::experimental::alignment<256>>>;

struct MyIP {
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

The type of the kernel argument `a` is `annotated_arg` specified with properties `buffer location` and `alignment`. To test the IP component in the FPGA simulation, the SYCL host code is required to provide for the kernel arguments runtime values that match with the `buffer location` and `alignment`. The helper function `alloc_annotated` (in `fpga_tools` namespace) provides this functionality with code simplicity, as it takes the `annotated_arg` type as the only template parameter and returns an instance of the same `annotated_arg` type, as the following code snippet shows

```c++
#include "annotated_class_util.hpp"

constexpr int kN = 8;
annotated_arg_t array_a = fpga_tools::alloc_annotated<annotated_arg_t>(kN, q);

// Initialize the host array
...
// Launch the kernel
q.single_task(MyIP{array_a, kN}).wait();
```

> **Note**: "alloc_annotated" function only takes `annotated_arg` templated with **pointer types** as the valid template parameter.


## Building the `annotated_class_clean_coding` Sample
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

3. Compile the design with the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Compilation Type    | Command
   |:---                 |:---
   | FPGA Emulator       | `make fpga_emu`
   | Optimization Report | `make report`
   | FPGA Simulator      | `make fpga_sim`
   | FPGA Hardware       | `nmake fpga`

### On a Windows* System
This design uses CMake to generate a build script for  `nmake`.

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```

3. Compile the design with the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   | Compilation Type    | Command (Windows)
   |:---                 |:---
   | FPGA Emulator       | `nmake fpga_emu`
   | Optimization Report | `nmake report`
   | FPGA Simulator      | `nmake fpga_sim`
   | FPGA Hardware       | `nmake fpga`

## Run the `annotated_class_clean_coding` Executable

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
   ```
   ./annotated_coding.fpga_emu
   ```

2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./annotated_coding.fpga_sim
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
   ```
   annotated_coding.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   annotated_coding.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

## Example Output

```
Running on device: Intel(R) FPGA Emulation Device
using aligned_alloc_shared to allocate a block of shared memory
using fpga_tools::alloc_annotated to allocate a block of shared memory
PASSED: all kernel results are correct
```

## License

Code samples are licensed under the MIT license. See
[License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
