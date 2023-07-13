# `Host Pipes` Sample

This FPGA sample is a tutorial that demonstrates how to use pipes to send and receive data between a host and a device.

| Area                  | Description
|:--                    |:--
| What you will learn   | Basics of host pipe declaration and usage
| Time to complete      | 30 minutes
| Category              | Concepts and Functionality

## Purpose

Pipes are a first-in first-out (FIFO) buffer construct that provides links between elements of a design. Access pipes through read and write application programming interfaces (APIs), without the notion of a memory address or pointer to elements within the FIFO.

Pipes connecting a host and a device are called host pipes. Use host pipes to move data between the host part of a design and a kernel that resides on the FPGA. A read and write API imposes FIFO ordering on accesses to this data. The advantage to this approach is that you do not need to write code to address specific locations in these buffers when accessing the data. Host pipes provide a "streaming" interface between host and FPGA, and are best used in designs where random access to data is not needed or wanted.

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

This tutorial illustrates some key concepts:

- Basics of declaring host pipes.
- Using blocking read and write API for host pipes.

### Declaring a Host Pipe

Each individual host pipe is a function scope class declaration of the templated pipe class. The first template parameter should be a user-defined type that differentiates this particular pipe from the others. The second template parameter defines the datatype of each element carried by the pipe. The third template parameter defines the pipe capacity, which is the guaranteed minimum number of elements of datatype that can be held in the pipe. In other words, for a given pipe with capacity `c`, the compiler guarantees that operations on the pipe will not block due to capacity as long as, for any consecutive `n` operations on the pipe, the number of writes to the pipe minus the number of reads does not exceed `c`.

```c++
// unique user-defined types
class FirstPipeT;
class SecondPipeT;

// two host pipes
using FirstPipeInstance = cl::sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    FirstPipeT, // An identifier for the pipe
    int,        // The type of data in the pipe
    8           // The capacity of the pipe
    >;
using SecondPipeInstance = cl::sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    SecondPipeT, // An identifier for the pipe
    int,         // The type of data in the pipe
    4            // The capacity of the pipe
    >;
```

In this example, `FirstPipeT` and `SecondPipeT` are unique user-defined types that identify two host pipes. The first host pipe (which has been aliased to `FirstPipeInstance`), carries `int` type data elements and has a capacity of `8`. The second host pipe (`SecondPipeInstance`) carries `float` type data elements, and has a capacity of `4`. Using aliases allows these pipes to be referred to by a shorter and more descriptive handle, rather than having to repeatedly type out the full namespace and template parameters.

#### Host pipe properties

Host pipes use a fourth template parameter beyond the three described earlier. This template parameter uses the oneAPI properties class to allow users to define additional semantic properties for a host pipe. The use of these properties is beyond the scope of this tutorial. You can find their definitions and usage in the *[Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/)*. Omitting the properties parameter (as has been done for the host pipes in this code sample) gives the host pipe the default values for these properties as described in the guide.

### Host Pipe API

Host Pipes expose read and write interfaces that allow a single element to be read or written in FIFO order to the pipe. These read and write interfaces are static class methods on the templated classes described in the Declaring a Host Pipe section above, and are described below.

Host pipes expose read and write interfaces that allow a single element to be read or written in FIFO order to the pipe. These read and write interfaces are static class methods on the templated classes that are described in the [Declaring a Host Pipe](#declaring-a-host-pipe) section. The API provides the following interfaces:

  - [Blocking write interface](#blocking-write)
  - [Non-blocking write interface](#non-blocking-write)
  - [Blocking read interface](#blocking-read)
  - [Non-blocking read interface](#non-blocking-read)

#### Blocking Write

The host pipe write interface writes a single element of the given datatype (`int` in the examples below) to the host pipe. On the host side, this class method takes a SYCL* device queue argument as its first argument, and the element being written as its second argument.

```c++
queue q(...);
...
int data_element = ...;

// blocking write from host to pipe
FirstPipeInstance::write(q, data_element);
```

In the FPGA kernel, writes to a host pipe take a single argument, which is the element being written.

```c++
float data_element = ...;

// blocking write from device to pipe
SecondPipeInstance::write(data_element);
```

#### Non-Blocking Write

Non-blocking writes add a `bool` argument in both host and device APIs that is passed by reference and returns true in this argument if the write was successful, and false if it was unsuccessful.

On the host:

```c++
queue q(...);
...
int data_element = ...;

// variable to hold write success or failure
bool success = false;

// attempt non-blocking write from host to pipe until successful
while (!success) FirstPipeInstance::write(q, data_element, success);
```

On the device:

```c++
float data_element = ...;

// variable to hold write success or failure
bool success = false;

// attempt non-blocking write from device to pipe until successful
while (!success) SecondPipeInstance::write(data_element, success);
```

#### Blocking Read

The host pipe read interface reads a single element of given datatype from the host pipe. Similar to write, the read interface on the host takes a SYCL* device queue as a parameter. The device read interface consists of the class method read call with no arguments.

On the host:

```c++
// blocking read in host code
float read_element = SecondPipeInstance::read(q);
```

On the device:

```c++
// blocking read in device code
int read_element = FirstPipeInstance::read();
```

#### Non-Blocking Read

Similar to non-blocking writes, non-blocking reads add a `bool` argument in both host and device APIs that is passed by reference and returns true in this argument if the read was successful, and false if it was unsuccessful.

On the host:

```c++
// variable to hold read success or failure
bool success = false;

// attempt non-blocking read until successful in host code
float read_element;
while (!success) read_element = SecondPipeInstance::read(q, success);
```

On the device:

```c++
// variable to hold read success or failure
bool success = false;

// attempt non-blocking read until successful in device code
int read_element;
while (!success) read_element = FirstPipeInstance::read(success);
```

### Host Pipe Connections

Host pipe connections for a particular host pipe are inferred by the compiler from the presence of read and write calls to that host pipe in your code. A host pipe can be connected from the host only to a single kernel. That is, host pipe calls for a particular host pipe must be restricted to the same kernel. Host pipes can also only operate in one direction. That is, host-to-kernel or kernel-to-host. Host code for a particular host pipe can contain either only all writes or only all reads to that pipe, and the corresponding kernel code for the same host pipe can consist only of the opposite transaction.

### Testing the Tutorial

In `hostpipes.cpp`, two host pipes are declared for transferring host-to-device data (`H2DPipe`) and device-to-host data (`D2HPipe`).

```c++
using H2DPipe = cl::sycl::ext::intel::experimental::pipe<H2DPipeID, ValueT, kPipeMinCapacity>;
using D2HPipe = cl::sycl::ext::intel::experimental::pipe<D2HPipeID, ValueT, kPipeMinCapacity>;
```

These host pipes are used to transfer data to and from `SubmitLoopBackKernel`, which reads a data element from the H2DPipe (parameterized in the kernel template as `InHostPipe`), processes it using the `SomethingComplicated()` function (a placeholder example of offload computation), and writes it back to the host via `D2HPipe` (template parameter `OutHostPipes`).

```c++
template<typename KernelId, typename InHostPipe, typename OutHostPipe>
event SubmitLoopBackKernel(queue& q, size_t count) {
  return q.single_task<KernelId>([=] {
    for (size_t i = 0; i < count; i++) {
      auto d = InHostPipe::read();
      auto r = SomethingComplicated(d);
      OutHostPipe::write(r);
    }
  });
}
```

The SubmitLoopBackKernel is exercised in two different ways: an alternating read/write test, and a launch-collect test. In the former case, the host writes an element to be processed into the H2DPipe, and immediately attempts to read this result from the D2HPipe. When this read is successful, the next iteration of the loop can proceed to write the next element to be processed to the H2DPipe. This minimizes the capacity needed for both host pipes, as each pipe will hold at most one element at a time.
The `SubmitLoopBackKernel` is exercised in two different ways: an alternating read/write test and a launch-collect test. In the former case, the host writes an element to be processed into `H2DPipe`, and immediately attempts to read this result from `D2HPipe`. When this read is successful, the next iteration of the loop can proceed to write the next element to be processed to `H2DPipe`. This configuration minimizes the capacity needed for both host pipes, as each pipe holds at most one element at a time.

```c++
  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": " << "Doing " << count << " writes & reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      H2DPipe::write(q, in[i]);
      out[i] = D2HPipe::read(q);
    }
  }
```

In the latter launch-collect test, the entire contents of the `in` vector are written to `H2DPipe` before the results are read from `D2HPipe`. For a pipelined kernel, this sequence has the advantage of pipeline parallelizing the offloaded computation on each input data element. However, this sequence can increase the capacity requirements for `H2DPipe`, `D2HPipe` or both. Since all of the input data elements are written to `H2DPipe` before any are read out of `D2HPipe`, the total capacity of the two pipes plus the kernel datapath must be greater than the total number of input elements.

```c++
  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": " << "Doing " << count << " writes" << std::endl;
    for (size_t i = 0; i < count; i++) {
      H2DPipe::write(q, in[i]);
    }

    std::cout << "\t " << r << ": " << "Doing " << count << " reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      out[i] = D2HPipe::read(q);
    }
  }
```

## Build the `Host Pipes` Tutorial

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
   > This tutorial only uses the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulates an FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the HTML optimization reports. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
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
   > This tutorial only uses the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      nmake report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced problem size).
      ```
      nmake fpga_sim
      ```
   4. Compile and run on FPGA hardware (longer compile time, targets an FPGA device).
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

#### Read the Reports

1. Locate `report.html` in the `hostpipes_report.prj/reports/` directory.
2. Open the **Views** menu and select **System Viewer**.
3. In the left-hand pane, select **LoopBackKernelID** under the System hierarchy.

In the main **System Viewer** pane, the pipe read and pipe write for the kernel are highlighted in the **LoopBackKernelID.B1** block. Selecting **LoopBackKernelID.B1** in the left-hand pane gives an expanded view of this block in the main pane, with the pipe read represented by a 'RD' node, and pipe write as a 'WR' node. Clicking on either of these nodes gives further information for these pipes in the **Details** pane. This pane will show that the read is reading from the `H2DPipeID` host pipe, and that the write is writing to the `D2HPipeID` host pipe, as well as verifying that both pipes have a width of 32 bits (corresponding to the `int` type) and depth of 8 (which is the `kPipeMinCapacity` that each pipe was declared with).

```c++
// forward declare kernel and pipe names to reduce name mangling
...
class H2DPipeID;
class D2HPipeID;
...
using H2DPipe = cl::sycl::ext::intel::experimental::pipe<
   // Usual pipe parameters
   H2DPipeID,         // An identified for the pipe
   ...
   >;

using D2HPipe = cl::sycl::ext::intel::experimental::pipe<
   // Usual pipe parameters
   D2HPipeID,         // An identified for the pipe
   ...
   >;
```

## Run the `Host Pipes` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./hostpipes.fpga_emu
   ```
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./hostpipes.fpga_sim
   ```
> **Note**: Running this sample on an actual FPGA device requires a BSP that supports host pipes. As there are currently no commercial BSPs with such support, only the IP Authoring flow is enabled for this code sample.
	
### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   hostpipes.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   hostpipes.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
> **Note**: Running this sample on an actual FPGA device requires a BSP that supports host pipes. As there are currently no commercial BSPs with such support, only the IP Authoring flow is enabled for this code sample.

## Example Output

```
Running Alternating write-and-read
	 Run Loopback Kernel on FPGA
	 0: Doing 16 writes & reads
	 1: Doing 16 writes & reads
	 2: Doing 16 writes & reads
	 Done

Running Launch and Collect
	 Run Loopback Kernel on FPGA
	 0: Doing 8 writes
	 0: Doing 8 reads
	 1: Doing 8 writes
	 1: Doing 8 reads
	 2: Doing 8 writes
	 2: Doing 8 reads
	 Done

PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
