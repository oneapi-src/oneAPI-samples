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
| Time to complete                  | 45 minutes



## Purpose
When optimizing a design, choosing the right interface can effectively improve the Quality of Results (QoR), often without requiring changes to the design's algorithm. Sometimes the system that a  will be added to, dictates what interfaces must be used.

This tutorial shows how to use the Memory Mapped Host (mmhost) macros to configure memory-mapped interfaces for kernel's arguments. These generated interfaces will follow the Avalon MM protocol. 

Deciding what combinations of Avalon-MM interfaces your kernal uses is dependent on both 
the desired area and performance of the kernal, as well as constraints from the system 
(i.e., what type of memory is available?, is there contention on the memory bus?, etc.).

### Memory-mapped interface parameters

The following parameters are available for configuration:

| Memory Attribute                 | Description               | Default Value
---                                |---   |---
| `buffer_location`          | The address space of the interface that associates with the host. | 0
| `awidth`          | The width of the memory-mapped address bus in bits. | 41
| `dwidth`          | The width of the memory-mapped data bus in bits. | 64
| `latency`         | The guaranteed latency from when a read command exits the kernal when the external memory returns valid read data. | 16
| `readwrite_mode`  | The port direction of the interface. (0: Read & Write, 1: Read only, 2: Write only) | 0
| `maxburst`        | The maximum number of data transfers that can associate with a read or write transaction. | 1
| `align`           | The alignment of the base pointer address in bytes. | 0
| `waitrequest`     | Adds the waitrequest signal that is asserted by the agent when it is unable to respond to a read or write request. | 0

### Default interface

If no memory-mapped interface macro is specified for a pointer kernel argument, then a mmhost interface with all default parameters is generated for it.
```c++
struct PointerIP{
  int *x, *y, *z;
  int size;
  
  PointerIP(int *x_, int *y_, int *z_, int size_)
    : x(x_), y(y_), z(z_), size(size_) {}

    void operator()() const {
      for (int i = 0; i < size; ++i) {
        int mul = x[i] * y[i];
        int add = mul + z[i];
        y[i] = mul;
        z[i] = add;
      }
    }
};
```

### Memory-mapped interface implementation restriction

The mmhost macros are restricted to use with functors, which means that they cannot be used when the kernel is written as a lambda function. Also, all arguments of the macro must be specified, even if just one parameter is being changed from the default values.

#### Example 1: A kernel expressed using a functor model with mmhost macro.
```c++
struct MyIP {
  mmhost(
    1,       // buffer_location or aspace
    28,      // address width
    64,      // data width
    16,      // latency
    1,       // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
    1,       // maxburst
    0,       // align, 0 defaults to alignment of the type
    1        // waitrequest, 0: false, 1: true
  ) int *my_pointer;

  MyIP(int *input_pointer) : my_pointer(input_pointer) { ... } // Constructor for our IP.
    ...
  }

  void operator()() const {
    ...
    // Kernel code
    my_pointer[x] = y;
    ...
  }
}
```



An Avalaon bus is generate for every unique global memory in the system. 

If the mmhost argument is specified as a conduit interface then an input wire is generated at the top-level device image IP to carry the the pointer address into the kernal. 

If the mmhost argument is specified as a register-mapped argument, the input pointer is written into a register map associated with the kernal. An Avalon bus is generated for writing those arguments into a register map, with no seperate wire needed at the top level device image to input the address.
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

If unspecified, when using only the ```mmhost(...)``` macro, a register_map based input argument interface is used.

### Memory Base Addresses assigned by the compiler

The compiler infers a unique global memory for each pointer argument that is annotated with a buffer location. An entry describing the interface of each such memory can be found in the file ipinterfaces.xml which is generated in the "*.prj" output directory. The entry specifies various properties of the memory including the start address that was assigned to the memory by the compiler. Each buffer_location will have its address defined there, among other parameters. 

To understand how these start addresses are generated and how the compiler assigns special values to the top bits of the addresses please see the complete documentation on memory-mapped interfaces. If the parameters are not set correctly, this can lead to undefined behaviour.

The file will have the following entries based on the kernel arguments used in this tutorial:

Example of a pointer argument in the ipinterfaces.xml:
```xml
<global_mem name="1" default="1" max_bandwidth="0" config_addr="0x0" type="device private" allocation_type="host,shared">
  <interface name="1" type="agent" width="64" address="0x2000000000000" latency="16" latency_type="fixed" waitrequest="1" size="0x10000000">
    <port name="mem1_r" direction="r"/>
  </interface>
</global_mem>
<global_mem name="2" max_bandwidth="0" config_addr="0x0" type="device private" allocation_type="host,shared">
  <interface name="2" port="mem2_rw" type="agent" width="64" address="0x3000000000000" latency="16" latency_type="fixed" waitrequest="1" size="0x10000000"/>
</global_mem>
<global_mem name="3" max_bandwidth="0" config_addr="0x0" type="device private" allocation_type="host,shared">
    <interface name="3" port="mem3_rw" type="agent" width="64" address="0x4000000000000" latency="16" latency_type="fixed" waitrequest="1" size="0x10000000"/>
</global_mem>
```

### Avalon MMHost Bus Signals
A separate Avalon signal bus is generated at the top-level device image for each kernel argument that has a unique buffer location specified for it. The signal bus has customized address bus size, data bus size etc. based on the properties specified in the C++ source file.

A simplified description of the signals that constitute the Avalon signal bus is shown below. A detailed description can be found in the Avalon Interface Specifications Document (ID: 683091, release 22.3) in section 3.2 titled: Avalon Memory Mapped Interface Signal Roles: 

| Signal | Description | Width
| --- | --- | ---
| `address` | Represents the byte address | 1 - 64 bits
| `writedata` | Data for write transfers | 8, 16, 32, 64, 128, 256, 512, or 1024 bits
| `readdata` | The readdata driven from the agent to the host in response to a read transfer | 8, 16, 32, 64, 128, 256, 512, or 1024 bits
| `byteenable` | Enables one or more specific byte lanes during transfers on interfaces of width greater than 8 bits. Each bit in byteenable corresponds to a byte in writedata and readadata | 2, 4, 8, 16, 32, 64, or 128 bits
| `waitrequest` | An agent asserts this waitrequest signal when unable to respond to read/write requests | 1 bit 
| `readdatavalid` | Used for variable-latency pipelined read transfers, asserts that the readdata signal contains valid data | 1 bit
| `burstcount` | Used to indicate the number of transfers in each burst | 1 - 11 bits

### Buffer Location in Pointer Address Bits
The compiler encodes certain information regarding the virtual address space in top bits of the 64 bits wide pointer address. Below is a table showing which bits correspond to what function

| Bits | Function
| --- | ---
| 0 - 40 | Addressing within the memory system
| 41 - 46 | Buffer Location information
| 47 | Not Used
| 48 - 61 | Used by runtime layer
| 62 - 63 | Indicates whether a pointer is a global or local memory pointer

Users do not need to encode this information themselves in most cases. The compiler automatically generates logic to embed this information from the buffer location specified on the pointer kernel argument in the source file.

Why is this embedding needed?

In some cases, the compiler cannot figure out which buffer location a pointer corresponds to, and it creates logic in the generated RTL that inspects the top bits of the pointer at runtime to detect the buffer location and route the memory transaction to the correct global memory



### Tutorial Code Overview
This tutorial demonstrates how to implement a memory-mapped interface in an IP authoring flow. The design performs a multiply and add operation on two input vectors, stores the result of the multiple in the second vector, and the result of the add in an output-only vector.

* Memory-mapped interfaces needs to contain all parameters to work correctly.

## Testing the Tutorial
In ```mmhost.cpp```, two seperate Kernal IP's are declared for the same operation of adding and multiplying two vectors. ```PointerIP``` which declares a global memory interface for all pointers, and ```VectorMADIP``` which declares a seperate interface buffer location to each argument. 
```c++
struct PointerIP {
  int *x, *y, *z;
  int size;
  
  PointerIP(int *x_, int *y_, int *z_, int size_)
    : x(x_), y(y_), z(z_), size(size_) {}

    void operator()() const {
      ...
    }
};

struct VectorMADIP {
  mmhost(BL1, ...) int *x;
  mmhost(BL2, ...) int *y;
  mmhost(BL3, ...) int *z;
  int size;

  VectorMADIP(int *x_, int *y_, int *z_, int size_)
      : x(x_), y(y_), z(z_), size(size_) {}

    void operator()() const {
      ...
    }
```

We test and compare the kernal times for both of these IP's. Notice that the ```VectorMADIP``` is much faster. This is because all three pointers can be accessed concurrently without arbitration. 

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

## Examining the Reports
Locate `report.html` in the `mmhost_report.prj/reports/` directory. Open the report in Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the Area Analysis section of the optimization report. The Kernel System section displays the area consumption of each kernel. Notice that the `VectorMADIP` kernal consumes way less area under all categories than the `PointerIP` kernal. This is due to stall free memory accesses and the removal of arbiration logic, both of which come from separating the accesses into their own interfaces.

Navigate to the Loop Throughput section under Throughput Analysis, and you will see that the `VectorMADIP` Kernal has a lower latency than the `PointerIP` Kernal, and there are less blcoks being scheduled. This is because the kernal has access to all 3 memories in parallel without contention.


## Running the Sample

1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./mmhost.fpga_emu     (Linux)
     mmhost.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA simulator device (the kernel executes on the CPU):
  * On Linux
    ```bash
    CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./loop_fusion.fpga_sim
    ```
  * On Windows
    ```bash
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
    loop_fusion.fpga_sim.exe
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
    ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`):
     ```
     ./loop_fusion.fpga         (Linux)
     loop_fusion.fpga.exe       (Windows)
     ```


### Example of Output

```
MMHost kernel time : 7742.1 ms
Pointer kernel time : 58357.4 ms
elements in vector : 10000
--> PASS
```

