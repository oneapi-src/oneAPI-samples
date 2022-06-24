# Memory-Mapped Interfaces (mmhost)
This FPGA tutorial demonstrates how to use the memory-mapped interfaces with IP authoring and set the configurable parameters.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               |  The basic concepts of on-chip memory attributes <br> How to apply memory attributes in your program <br> How to confirm that the memory attributes were respected by the compiler <br> A case study of the type of performance/area trade-offs enabled by memory attributes
| Time to complete                  | 30 minutes



## Purpose
With the mmhost interface controls, the user can configure memory-mapped interfaces for their components.

### Memory-mapped interface parameters

The following parameters are available for configuration:

| Memory Attribute                 | Description
---                                |---
| `aspace`          | The address space of the interface that associates with the host.
| `awidth`          | The width of the memory-mapped data bus in bits
| `dwidth`          | The width of the memory-mapped address bus in bits.
| `latency`         | The guaranteed latency from when a read command exits the component when the external memory returns valid read data.
| `readwrite_mode`  | The port direction of the interface. (0: Read & Write, 1: Read only, 2: Write only)
| `maxburst`        | The maximum number of data transfers that can associate with a read or write transaction.
| `align`           | The alignment of the base pointer address in bytes. 
| `waitrequest`     | Adds the waitrequest signal that is asserted by the agent when it is unable to respond to a read or write request.

### Default interface

If no memory-mapped interface parameters is informed for a pointer, then a mmhost interface with all default parameters is assumed for it.

### Memory-mapped interface implementation restriction

Memory-mapped interface is restricted to function implementations, which means that it is not supported for lambda implementation. Also, all parameters must be preset, even if just one parameter is being changed from the default values.

#### Example Functor
```c++
struct MyIP {
  mmhost(
    ... // All mmhost parameters.
  ) int *my_pointer;

  MyIP(int *input_pointer) : my_pointer(input_pointer) { ... } // Constructor for our IP.
    ...
  }

  void operator()() const {
    ...
    // Functor code
    my_pointer[x] = y;
    ...
  }
}
```


#### Example 1: How to set-up a correct mmhost interface
```c++
// A memory-mapped interface must contain all parameters, in the following order:
  mmhost(
    1,       // buffer_location or aspace
    28,      // address width
    64,      // data width
    16,      // latency
    1,       // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
    1,       // maxburst
    0,       // align, 0 defaults to alignment of the type
    1        // waitrequest, 0: false, 1: true
  ) int *memory_mapped_pointer;
```

#### Example 2: Changing default mmhost interface.
```c++
// A memory-mapped interface that is implemented as a register.
  register_map_mmhost(
    1,       // buffer_location or aspace
    28,      // address width
    64,      // data width
    16,      // latency
    1,       // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
    1,       // maxburst
    0,       // align, 0 defaults to alignment of the type
    1        // waitrequest, 0: false, 1: true
  ) int *memory_mapped_pointer;
// A memory-mapped interface that is implemented as a wire.
  conduit_mmhost(
    1,       // buffer_location or aspace
    28,      // address width
    64,      // data width
    16,      // latency
    1,       // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
    1,       // maxburst
    0,       // align, 0 defaults to alignment of the type
    1        // waitrequest, 0: false, 1: true
  ) int *memory_mapped_pointer;
```

### Tutorial Code Overview
This tutorial demonstrates how to implement a memory-mapped interface in an IP authoring flow. The design performs a multiply and add operation on two input vectors, stores the result of the multiple in the second vector, and the result of the add in an output-only vector.

* Memory-mapped interfaces needs to contain all parameters to work correctly.
* Available parameters and default values can be fount at the header <sycl/ext/intel/experimental/interfaces.hpp>

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `mmhost` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

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

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Generate the optimization report:
     ```
     make report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```

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
   * Generate the optimization report:
     ```
     nmake report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./mmhost.fpga_emu     (Linux)
     mmhost.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./mmhost.fpga         (Linux)
     ```

### Example of Output

```
kernel time : 726.779 ms
elements in vector : 15
Passed correctness check
```


