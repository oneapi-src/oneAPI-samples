# `Streaming Interfaces` Sample

This FPGA sample is a tutorial that demonstrates how to use pipes to implement streaming interfaces on IP components. If you have not already gone over the IP Authoring Interfaces Overview Tutorial, it is recommended that you do so before continuing with this tutorial.

| Area                  | Description
|:--                    |:--
| What you will learn   | How to use pipes to implement streaming interfaces on IP components
| Time to complete      | 30 minutes
| Category              | Concepts and Functionality

## Purpose

Pipes are a first-in first-out (FIFO) buffer construct that provide links between elements of a design. They are accessed through read and write APIs without the notion of a memory address or pointer to elements within the FIFO.

The concept of a pipe provides us with a mechanism for specifying and configuring streaming interfaces on an IP component. This tutorial will demonstrate how to declare and configure a pipe to implement a streaming interface on an IP component.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

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

### Configuring a Pipe to Implement a Streaming Interface

Each individual pipe is a function scope class declaration of the templated `pipe` class.
- The first template parameter is a user-defined type that differentiates this particular pipe from the others and provides a name for the interface in the  RTL.
- The second template parameter defines the datatype of elements carried by the interface.
- The third template parameter allows you to optionally specify a non-negative integer representing the capacity of the buffer on the input. This can help avoid some amount of bubbles in the pipeline in case the component itself stalls.
- The fourth template parameter uses the oneAPI properties class to allow users to optionally define additional semantic properties for a pipe. **The `protocol` property is what will allow us to configure a pipe to implement a streaming interface.** A list of those properties relevant to this sample is given in Table 1 (please note that this table is *not* complete; see the [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/optimization-guide/current/host-pipe-declaration.html) for more information on how to use pipes in other applications).

#### Table 1. Properties used to Configure a Pipe to Implement a Streaming Interface

| Property | Valid Values | Default Value |
| ---------| ------------ | ------------- |
| `ready_latency<int>`                    | non-negative integer | 0 |
| `bits_per_symbol<int>`                  | non-negative integer that divides the size of the data type | 8 |
| `uses_valid<bool>`                      | boolean      | `true` |
| `first_symbol_in_high_order_bits<bool>` | boolean      | `true` |
| `protocol`                              | `protocol_avalon_streaming` / `protocol_avalon_streaming_uses_ready` | `protocol_avalon_streaming_uses_ready` |

> **Note:** These properties may be specified in *any* order within the oneAPI `properties` object. Omitting a single property from the properties class instructs the compiler to assume the default value for that property (i.e., you can just define the properties you would like to change from the default). Omitting the properties template parameter entirely instructs the compiler to assume the default values for *all* properties.

#### Example 1.

The following example declares a pipe to implement a streaming interface with  `ready` and `valid` signals, using the defaults for all parameters.

```c++
// Unique user-defined types
class FirstPipeT;

// Pipe properties (listed here are the defaults; this achieves the same
// behavior as not specifying any of these properties)
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>),
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready
);

using FirstPipeInstance = sycl::ext::intel::experimental::pipe<
    FirstPipeT,      // An identifier for the pipe
    int,             // The type of data in the pipe
    8,               // The capacity of the pipe
    PipePropertiesT  // Customizable pipe properties
    >;
```

### Avalon Streaming Sideband Signals

You can enable Avalon streaming sideband signal support by using the special `StreamingBeat` struct provided by the `pipes_ext.hpp` header file (`sycl/ext/intel/prototype/pipes_ext.hpp`) as the data type to your pipe. Only the `StreamingBeat` struct generates sideband signals when used with a pipe.

The `StreamingBeat` struct is templated on three parameters.
- The first template parameter defines the type of the data being communicated through the interface.
- The second parameter is used to enable additional `start_of_packet` (`sop`) and `end_of_packet` (`eop`) signals to the Avalon interface.
- The third template parameter is used to enable the `empty` signal, which indicates the number of symbols that are empty during the `eop` cycle.

#### Example 2.

The following example shows how to configure a `StreamingBeat` struct to use `sop`, `eop` and `empty` by setting the second and third template parameters to `true`. Using the `StreamingBeatT` type defined here in a pipe will result in additional `startofpacket`, `endofpacket`, and `empty` signals on the interface.

```c++
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

...

using StreamingBeatDataT = sycl::ext::intel::experimental::StreamingBeat<unsigned char, true, true>;

...

StreamingBeatT out_beat(data, sop, eop, empty);
SecondPipeInstance::write(out_beat);
```

### Read and Write APIs

The read and write APIs are the same as for all pipes. See the IP Authoring Interfaces Overview Tutorial for more information.

## Build the `Streaming Interfaces with Pipes` Tutorial

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
   > This tutorial is only intended for use in the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

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
   > This tutorial is only intended for use in the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

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
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Streaming Interfaces with Pipes` Tutorial

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

In `threshold_packets.cpp`, two pipes are declared for implementing the streaming input and streaming output interfaces on a kernel which thresholds pixel values in an image. The streams use start of packet and end of packet signals to determine the beginning and end of the image.


### Reading the Reports

After compiling in the report flow, locate and open the `report.html` file in the `threshold_packets.report.prj/reports/` directory. Under the `Threshold` kernel in the System Viewer, the streaming in and streaming out interfaces can be seen, shown by the pipe read and pipe write nodes respectively. Clicking on either of these nodes gives further information for these interfaces in the Details pane. This pane will identify that the read is coming from `InPixel`, and that the write is going to `OutPixel`, as well as verifying that both interfaces have a width of 24 bits (corresponding to the `StreamingBeatT` type) and depth of 8 (which is the capacity that each pipe was declared with).

<p align="center">
  <img src=assets/kernel.png />
</p>

### Viewing the Simulation Waveform

After compiling in the simulation flow and running the resulting executable, locate and run the `view_waveforms.sh` script in the `threshold_packets.fpga_sim.prj/` directory. Here you can see the `ready`, `valid` and `data` signals of the streaming input and streaming output interfaces (`InPixel` and `OutPixel` respectively). You can also see the `startofpacket` and `endofpacket` sideband signals that were added to the interface.

<p align="center">
  <img src=assets/sim_waveform.png />
</p>

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
