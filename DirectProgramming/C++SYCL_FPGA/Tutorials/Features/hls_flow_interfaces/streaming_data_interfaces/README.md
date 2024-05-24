# `Streaming Data Interfaces` Sample

This FPGA sample is a tutorial that demonstrates how to implement streaming data interfaces on an IP component. It is recommended that you review the [Component Interfaces Comparison](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/hls_flow_interfaces/component_interfaces_comparison) tutorial before continuing with this one.

| Area                  | Description
|:--                    |:--
| What you will learn   | How to use pipes to implement streaming data interfaces on an IP component
| Time to complete      | 30 minutes
| Category              | Concepts and Functionality

## Purpose

Pipes are a first-in first-out (FIFO) buffer construct that provide data links between elements of a design. They are accessed through read and write APIs without the notion of a memory address or pointers to elements within the FIFO.

The concept of a pipe is an intuitive mechanism for specifying streaming data interfaces on an IP component. This tutorial demonstrates how to use the pipe API to configure an Avalon streaming interface.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware             | Intel® Agilex® 7, Agilex® 5, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note:** Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.

> **Warning:** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples. It is categorized as a Tier 2 sample that demonstrates a compiler feature.

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

### Configuring a Pipe to Implement a Streaming Data Interface

Each pipe is a class declaration of the templated `pipe` class. A pipe declaration takes two mandatory and two optional parameters, as summarized in Table 1.

#### Table 1. Template Parameters of the `pipe` Class

| Template Parameter                      | Description
| ---                                     | ---
| `name`                                  | A user-defined type that uniquely identifies this pipe. The name of this type is also used to identify the interface in the generated RTL.
| `dataT`                                 | The type of data that passes through the pipe*. This is the data type that is read during a successful pipe read() operation, or written during a successful pipe write() operation. The type must have a standard layout and be trivially copyable.
| `min_capacity`                          | User-defined minimum number of words (in units of `dataT`) that the pipe must be able to store without any being read out. This parameter is optional, and defaults to 0.
| `properties`                            | An unordered list of SYCL properties that define additional semantic properties for a pipe. This parameter is optional.**

> **Note**: Omitting a single property from the properties class instructs the compiler to assume the default value for that property, so you can just define the properties you would like to change from the default. Omitting the properties template parameter entirely instructs the compiler to assume the default values for all properties.

Below is a summary of all relevant SYCL properties which can be applied to a `pipe` using the `properties` template parameter. Please note that this table is not complete; see the [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/optimization-guide/current/host-pipe-declaration.html) for more information on how to use pipes in other applications.


#### Table 2. Properties used to Configure a Pipe to Implement a Streaming Data Interface

| Property                                | Default Value                          | Valid Values
| ---                                     | ---                                    | ---
| `ready_latency<int>`                    | 0                                      | non-negative integer
| `bits_per_symbol<int>`                  | 8                                      | non-negative integer that divides the size of the data type
| `uses_valid<bool>`                      | `true`                                 | boolean
| `first_symbol_in_high_order_bits<bool>` | `true`                                 | boolean
| `protocol`                              | `protocol_avalon_streaming_uses_ready` |`protocol_avalon_streaming` / `protocol_avalon_streaming_uses_ready`

See [this page](https://www.intel.com/content/www/us/en/docs/programmable/683091/current/st-interface-properties.html) for more information on the Avalon interface specifications.

#### Example 1.

The following example explicitly declares a pipe that implements a streaming data interface with  `ready` and `valid` signals, using the defaults for all other parameters.

```c++
// Forward declare the pipe name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class FirstPipeT;

// Pipe properties (listed here are the defaults; this achieves the same
// behavior as not specifying any of these properties)
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready));

using FirstPipe = sycl::ext::intel::experimental::pipe<
    FirstPipeT,      // An identifier for the pipe
    int,             // The type of data in the pipe
    8,               // Minimum capacity of the pipe (buffer depth)
    PipePropertiesT  // Customizable pipe properties
    >;
```

### Avalon Streaming Sideband Signals

Pipes support a subset of Avalon streaming sideband signals. You can add these to your pipe interface by using the special `StreamingBeat` structure provided by the `pipes_ext.hpp` header file (`sycl/ext/intel/prototype/pipes_ext.hpp`) as the `dataT` of your pipe. Only the `StreamingBeat` structure generates sideband signals when used with a pipe.

The `StreamingBeat` structure is templated on three parameters, as summarized in Table 3.

#### Table 3. Template Parameters of the `StreamingBeat` Structure

| Template Parameter                      | Description
| ---                                     | ---
| `dataT`                                 | The datatype of elements carried by the `data` signal of the Avalon streaming interface.
| `uses_packets`                          | A boolean that indicates whether to enable the `startofpacket` (`sop`) and `endofpacket` (`eop`) sideband signals on the Avalon streaming interface.
| `uses_empty`                            | A boolean that indicates whether to enable the `empty` sideband signal on the Avalon streaming interface

#### Example 2.

The following example shows how to configure a `StreamingBeat` structure to use `sop`, `eop` and `empty` by setting the second and third template parameters to `true`. Using the `StreamingBeatT` type defined here in a pipe will result in `startofpacket`, `endofpacket`, and `empty` signals appearing on the resulting Avalon streaming interface.

```c++
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

...

using StreamingBeatDataT = sycl::ext::intel::experimental::StreamingBeat<unsigned short, true, true>;

...

StreamingBeatT out_beat(data, sop, eop, empty);
SecondPipeInstance::write(out_beat);
```

### Pipe API

Pipes expose read and write interfaces that allow a single element to be read or written in FIFO order to the pipe.

See the [Host Pipes](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/experimental/hostpipes) code sample for more details on the read and write APIs.

## Build the `Streaming Data Interfaces` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
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
   > This tutorial is only intended for use in the SYCL HLS flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the optimization report.
      ```
      make report
      ```
   3. Compile and run for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets an FPGA device).
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
   > This tutorial is only intended for use in the SYCL HLS flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report.
      ```
      nmake report
      ```
   3. Compile and run for simulation (fast compile time, targets simulated FPGA device).
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets an FPGA device).
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your 'build' directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory, for example:
>
>  ```
  > C:\samples\build> cmake -G "NMake Makefiles" C:\long\path\to\code\sample\CMakeLists.txt
>  ```
## Run the `Streaming Data Interfaces` Tutorial

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./threshold_packets.fpga_emu
   ```
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./threshold_packets.fpga_sim
   ```
	
### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   threshold_packets.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   threshold_packets.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

## Understanding the Tutorial

In `threshold_packets.cpp`, two pipes are declared to implement the input and output streaming interfaces on a kernel which thresholds pixel values in an image. The streams use `startofpacket` and `endofpacket` signals to determine the beginning and end of the image.


### Reading the Reports

After compiling the `report` target, locate and open the `report.html` file in the `threshold_packets.report.prj/reports/` directory. Under the `Threshold` kernel in the System Viewer, the streaming in and streaming out interfaces can be seen, shown by the pipe read and pipe write nodes respectively. Clicking on either of these nodes gives further information about these interfaces in the 'details' pane. The 'details' pane will identify that the read is coming from `InStream`, and that the write is going to `OutStream`, as well as verifying that both interfaces have a width of 32 bits (corresponding to size of the `StreamingBeatT` type) and depth of 8 (which is the capacity that each pipe was declared with).

<p align="center">
  <img src=assets/kernel.png />
</p>

### Viewing the Simulation Waveform

After compiling in the simulation flow and running the resulting executable, locate and run the `view_waveforms.sh` script in the `threshold_packets.fpga_sim.prj/` directory. Here you can see the `ready`, `valid` and `data` signals of the streaming input and streaming output interfaces (`InStream` and `OutStream` respectively). You can also see the `startofpacket` and `endofpacket` sideband signals that were added to the interface.

<p align="center">
  <img src=assets/sim_waveform.png />
</p>

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
