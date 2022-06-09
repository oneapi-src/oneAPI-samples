

# Host Pipes
This FPGA tutorial demonstrates how to use pipes to send and receive data between host and device. Pipes are a first in first out (FIFO) buffer construct that provide links between elements of a design. They are accessed through read and write application programming interfaces (APIs), without the notion of a memory address/pointer to elements within the FIFO. Pipes that connect between host and device are referred to as host pipes.


***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support (and SYCL USM support) <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Basics of host pipe declaration and usage 
| Time to complete                  | 30 minutes



## Purpose

Host pipes provide a way for users to move data between the host part of a design and a kernel residing on the FPGA. They provide this functionality through a read and write API that imposes FIFO ordering on accesses to this data. The advantage to such an approach is that it alleviates the need for a user to write code to address specific locations in these buffers when accessing the data. Host pipes provide a "streaming" interface between host and FPGA, best used in designs where random access to data is not required or warranted.

#### Prototype Implementation

The host pipe implementation in oneAPI 2022.3 is a prototype implementation that relies on experimental features that are not incorporated into the standard inter-kernel pipes already supported. To maintain separation from the existing inter-kernel pipe implementation, host pipes in this oneAPI version have been declared in a different namespace than inter-kernel pipes. This namespace is

```c++
cl::sycl::ext::intel::prototype
```

Additionally, the oneAPI 2022.3 prototype implementation of host pipes relies on Unified Shared Memory (USM). As such, only boards and devices that support USM can be used with host pipes in this release.

Both of these restrictions (separate namespace and requiring USM) will be lifted in a future release. 

### Declaring a Host Pipe
Each individual host pipe is a function scope class declaration of the templated pipe class. The first template parameter should be a user-defined type that will differentiate this particular pipe from all others. The second template parameter defines the datatype of each element carried by the pipe. The third template parameter defines the pipe capacity, which is the guaranteed minimum number of elements of datatype that can be held in a pipe. In other words, for a given pipe with capacity `c`, the compiler will guarantee that operations on the pipe will not block due to capacity as long as, for any consecutive `n` operations on the pipe, the number of writes to the pipe minus the number of reads does not exceed `c`.

```c++
// unique user-defined types
class FirstPipeT;
class SecondPipeT;

// two host pipes 
using FirstPipeInstance = cl::sycl::ext::intel::prototype::pipe<
    // Usual pipe parameters
    FirstPipeT, // An identifier for the pipe
    int,        // The type of data in the pipe
    8,          // The capacity of the pipe
    // Additional host pipe parameters
    kReadyLatency,                   // Latency for ready signal deassert
    kBitsPerSymbol,                  // Symbol size on data bus
    true,                            // Exposes a valid on the pipe interface
    false,                           // First symbol in high order bits
    protocol_name::AVALON_STREAMING  // Protocol
    >;
using SecondPipeInstance = cl::sycl::ext::intel::prototype::pipe<
    // Usual pipe parameters
    SecondPipeT, // An identifier for the pipe
    int,         // The type of data in the pipe
    4,           // The capacity of the pipe
    // Additional host pipe parameters
    kReadyLatency,                   // Latency for ready signal deassert
    kBitsPerSymbol,                  // Symbol size on data bus
    true,                            // Exposes a valid on the pipe interface
    false,                           // First symbol in high order bits
    protocol_name::AVALON_STREAMING  // Protocol
    >;
```

In this example, FirstPipeT and SecondPipeT are unique user-defined types that will idenfify two different host pipes. The first host pipe (which has been aliased to FirstPipeInstance), carries int type data elements and has a capacity of 8. The second host pipe (SecondPipeInstance) carries float type data elements, and has a capacity of 4. Additonal host pipe parameters beyond these first three have been set to sensible default values, but are not germane to this tutorial. Users may consult the [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) for a thorough description of these parameters. Using aliases allows these pipes to be referred to by a shorter and more descriptive handle, rather than requiring the full namespace and template parameters to be repeatedly written.

#### Additional template parameters

Host pipes use additional template parameters beyond the three described above. The use of these parameters is beyond the scope of this tutorial; their definitions and usage can be found in the [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide). Suitable values for these parameters consistent with non-specialized host pipe usage have been used in the accompanying tutorial code.


### Host Pipe API

Host Pipes expose read and write interfaces that allow a single element to be read or written in FIFO order to the pipe. These read and write interfaces are static class methods on the templated classes described in the Declaring a Host Pipe section above, and are described below.

#### Blocking Write

The host pipe write interface writes a single element of the given datatape (`int` in the examples below) to the host pipe. On the host side, this class method takes a SYCL device queue argument as its first argument, and the element being written as its second argument.

```c++
queue q(...);
...
int data_element = ...;

// blocking write from host to pipe
FirstPipeInstance::write(q, data_element);
```

In the FPGA kernel, writes to a host pipe take a single argument, which is the element being written.

```c++
int data_element = ...;

// blocking write from device to pipe
SecondPipeInstance::write(data_element);
```

#### Non-blocking Write

Non-blocking writes add a bool argument in both host and device APIs that is passed by reference and returns true in this argument if the write was successful, and false if it was unsuccessful. 

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
int data_element = ...;

// variable to hold write success or failur
bool success = false;

// attempt non-blocking write from device to pipe until successful
while (!success) SecondPipeInstance::write(data_element, success);
```

#### Blocking Read

The host pipe read interface reads a single element of given datatape from the host pipe. Similar to write, the read interface on the host takes a SYCL device queue as a parameter. The device read interface consists of the class method read call with no arguments. 

On the host:

```c++
// blocking read in host code
int read_element = SecondPipeInstance::read(q);
```

On the device:

```c++
// blocking read in device code
int read_element = FirstPipeInstance::read();
```

#### Non-blocking Read

Similar to non-blocking writes, non-blocking reads add a bool argument in both host and device APIs that is passed by reference and returns true in this argument if the read was successful, and false if it was unsuccessful. On the host:

```c++
// variable to hold read success or failure
bool success = false;

// attempt non-blocking read until successful in host code
int read_element;
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

### Host pipe connections

Host pipe connections for a particular host pipe are inferred by the compiler from the presence of read and write calls to that host pipe in user code. A host pipe may only be connected from the host to a single kernel, i.e., host pipe calls for a particular host pipe must be restricted to the same kernel. Host pipes may also only operate in one direction, i.e., host-to-kernel or kernel-to-host. This means that host code for a particular host pipe may only contain either all writes or all reads to that pipe, and the corresponding kernel code for the same host pipe may only consist of the opposite transaction.

### Testing the Tutorial
In `hostpipes.cpp`, two hostpipes are declared for transferring host-to-device data (`H2DPipe`) and device-to-host data (`D2HPipe`).
```c++
using H2DPipe = cl::sycl::ext::intel::prototype::pipe<H2DPipeID, ValueT, kPipeMinCapacity, kReadyLatency, kBitsPerSymbol, true, false, protocol_name::AVALON_STREAMING>;
using D2HPipe = cl::sycl::ext::intel::prototype::pipe<D2HPipeID, ValueT, kPipeMinCapacity, kReadyLatency, kBitsPerSymbol, true, false, protocol_name::AVALON_STREAMING>;
```
These host pipes are used to transfer data to and from the SubmitLoopBackKernel, which reads a data element from the H2DPipe (parameterized in the kernel template as InHostPipe), processes it using the something_complicated() function (a placeholder example of offload computation), and writes it back to the host via the D2HPipe (template parameter OutHostPipes).

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

The SubmitLoopBackKernel is exercised in two different ways: an alternating read-write test, and a launch-collect test. In the former case, the host writes an element to be processed into the H2DPipe, and immediately attempts to read this result from the D2HPipe. When this read is successful, the next iteration of the loop can proceed to write the next element to be processed to the H2DPipe. This minimizes the capacity needed for both host pipes, as each pipe will hold at most one element at a time.

```c++
  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": " << "Doing " << count << " writes & reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      H2DPipe::write(q, in[i]);
      out[i] = D2HPipe::read(q);
    }
  }
```

In the latter launch-collect test, the entire contents of the in vector are written to the H2DPipe, before the results are read from the D2HPipe. For a pipelined kernel, this has the advantage of pipeline parallelizing the offloaded computation on each input data element. However, this may increase the capacity requirements for the H2DPipe and/or the D2HPipe. Since all of the input data elements are written to the H2DPipe before any are read out of the D2HPipe, the total capacity of the two pipes plus the kernel datapath must be greater than the total number of input elements.

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

## Key Concepts
* Basics of host pipe declaration
* Usage of blocking read and write API for host pipes

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `hostpipes` Tutorial

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


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
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake ..
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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/hostpipes.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. 
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

>Note:* The Intel® FPGA PAC D5005 with Intel Stratix® 10 SX does not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs]([https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html))

## Examining the Reports
Locate `report.html` in the `hostpipes_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Open the Views menu and select System Viewer. In the left-hand pane, select LoopBackKernelID under the System hierarchy. In the main System Viewer pane, the pipe read and pipe write for the kernel are highlighted. They will show that the read is reading from the `cl::sycl::ext::intel::prototype::internal::pipe<detail::HostPipePipeId<H2DPipeID>` host pipe, and that the write is writing to the `cl::sycl::ext::intel::prototype::internal::pipe<detail::HostPipePipeId<D2HPipeID>` host pipe. Clicking on either of these host pipes will verify the width (32-bit corresponding to the `int` type) and depth (8, which is the kPipeMinCapacity each pipe was declared with).

You may notice that there are additional identifiers in the pipe template (notably, `detail::HostPipePipeId`). This is an internal implementation detail of this prototype feature. You can confirm the correspondence of these pipes to the ones declared in the source code by the template parameters `H2DPipeID` and `D2HPipeID`, which are the unique types declared in the source file and used in the pipe declarations:

```c++
// forward declare kernel and pipe names to reduce name mangling
...
class H2DPipeID;
class D2HPipeID;
...
using H2DPipe = cl::sycl::ext::intel::prototype::pipe<
   // Usual pipe parameters
   H2DPipeID,         // An identified for the pipe
   ...
   >;
   
using D2HPipe = cl::sycl::ext::intel::prototype::pipe<
   // Usual pipe parameters
   D2HPipeID,         // An identified for the pipe
   ...
   >;     
```

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./hostpipes.fpga_emu     (Linux)
     hostpipes.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./hostpipes.fpga         (Linux)
     ```

### Example of Output

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
