# `device_global` Sample

This sample is an FPGA tutorial that explains how to use `device_global` class as a way of keeping a state between multiple invocations of a kernel.

| Area                | Description
|:---                 |:---
| What you will learn | The basic usage of the `device_global` class <br> How to initialize a `device_global` to non-zero values<br>
| Time to complete    | 15 minutes
| Category            | Concepts and Functionality

## Purpose

This tutorial demonstrates a simple example of initializing a `device_global` class to a non-zero value, and how to use this approach to keep state between multiple re-launches of a kernel.

## Prerequisites

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware           | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software           | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

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
   style tier2 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.

## Key Implementation Details

This sample illustrates some key concepts.

- The basic usage of the `device_global` class
- How to initialize a `device_global` to non-zero value

### `device_global` Description

The `device_global` class is an extension that introduces device scoped memory allocations into SYCL that can be accessed within a kernel using syntax similar to C++ global variables; the class has unique instances per `sycl::device`. Similar to C++ global variables, a `device_global` variable have a namespace scope and is visible to all kernels within that scope.

A `device_global` class is instantiated from a class template. The template is parameterized by the type of the underlying allocation, and a list of properties. The type of the allocation also encodes the size of the allocation for potentially multidimensional array types. The list of properties change the functional behavior of the `device_global` instance to enable compiler and runtime optimizations. In the code sample two properties are used, `device_image_scope` and `host_access_none`. The `device_image_scope` property limits the scope of a single instance of a `device_global` from a device to a `device_image`. The absence of the `device_image_scope` property is not supported by the compiler. The `host_access` property is an assertion by the user telling the implementation whether the host code copies to or from the `device_global`. The property comes in four variants `host_access_none`, `host_access_read`, `host_access_write`, and `host_access_read_write`(the default), but only the `host_access_none` is supported currently. 

> **Note**: Further details on these and other properties can be found in the [Properties for `device_global` variables](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_device_global.asciidoc#properties-for-device-global-variables) section of the [SYCL `device_global` Language Specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_device_global.asciidoc). The language specification can also help you helps understand the API and of `device_globals` and restrictions on its usage.

A `device_global` instance can be used to store state across multiple relaunches of a kernel without having to pass in a `buffer` as a kernel argument. An example of an application that would benefit from such a state is where kernels are nodes in a state-machine.

### Initialize a `device_global` Instance

A `device_global` instance is always zero-initialized. If you need the first usage of a `device_global` instance in device code to be initialized to a non-zero value, use the  technique show in the code example below.

Instantiate a second `device_global<bool>` that represents a flag that controls when to initialize the `device_global` to a non-zero value. This flag is zero-initialized to `false`. Once initialization happens, the flag is set to `true` and initialization code does not execute on subsequent relaunches of the kernel.

```cpp
namespace exp = sycl::ext::oneapi::experimental;
using FPGAProperties = decltype(exp::properties(
    exp::device_image_scope, exp::host_access_none));
exp::device_global<int, FPGAProperties> val;
exp::device_global<bool, FPGAProperties> is_val_initialized;
int main () {
  sycl::queue q;
  q.submit([&](sycl::handler& h) {
    h.single_task([=] {
      // Initialization happens only once
      if (!is_val_initialized) {
        val.get() = 42;
        is_val_initialized.get() = true;
      }
      // uses of `val`
    });
  });
}
```

## Build the `device_global` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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

1. Change to the sample directory.
2. Build the program for Intel® Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulates an FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the HTML optimization reports. The report resides at `device_global_report.prj\reports\report.html`.
      ```
      make report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile and run on FPGA hardware (longer compile time, targets an FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Intel® Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report. The report resides at `device_global_report.a.prj\reports\report.html`.
      ```
      nmake report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced problem size).
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `device_global` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./device_global.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```bash
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./device_global.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./device_global.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   device_global.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   device_global.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   device_global.fpga.exe
   ```

## Example Output
```
PASSED: The results are correct
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).