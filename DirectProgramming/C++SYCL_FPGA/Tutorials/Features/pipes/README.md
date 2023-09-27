# `Pipes` Samples

This sample is an FPGA tutorial that shows how to use pipes to transfer data between kernels.

| Area                 | Description
|:--                   |:--
| What you will learn  | The basics of using SYCL*-compliant pipes extension for FPGA. <br> How to declare and use pipes. 
| Time to complete     | 15 minutes
| Category             | Concepts and Functionality

## Purpose

This tutorial demonstrates how a kernel in a SYCL*-compliant FPGA program transfers
data to or from another kernel using the pipe abstraction.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
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

> **Warning**: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

## Key Implementation Details

The sample illustrates the following important concepts.

- The basics of the SYCL*-compliant pipes extension for FPGA.
- How to declare and use pipes in a program.

### Definition of a Pipe

The primary goal of pipes is to allow concurrent execution of kernels that need
to exchange data. A pipe is a FIFO data structure connecting two endpoints that communicate
using the pipe's `read` and `write` operations. An endpoint can be either a kernel
or an external I/O on the FPGA. Therefore, there are three types of pipes:

- kernel-kernel
- kernel-I/O
- I/O-kernel

This tutorial focuses on kernel-kernel pipes, but
the concepts discussed here apply to other kinds of pipes as well.

The `read` and `write` operations have two variants:

- **Blocking variant**: Blocking operations may not return immediately but are always successful.
- **Non-blocking variant**: Non-blocking operations take an extra Boolean parameter
that is set to `true` if the operation happened successfully.

Data flows in a single direction inside pipes. In other words, for a pipe `P`
and two kernels using `P`, one of the kernels is exclusively going to perform
`write` to `P` while the other kernel is exclusively going to perform `read` from
`P`. Bidirectional communication can be achieved using two pipes.

Each pipe has a configurable `capacity` parameter describing the number of `write`
operations that may be performed without any `read` operations being performed. For example,
consider a pipe `P` with capacity 3, and two kernels `K1` and `K2` using
`P`. Assume that `K1` performed the following sequence of operations: `write(1)`, `write(2)`, `write(3)`

In this situation, the pipe is full because three (the `capacity` of
`P`) `write` operations were performed without any `read` operation. In this
situation, a `read` must occur before any other `write` is allowed.

If a `write` is attempted to a full pipe, one of two behaviors occurs:

- If the operation is non-blocking, it returns immediately, and its Boolean parameter is set to `false`. The `write` does not have an effect.
- If the operation is blocking, it does not return until a `read` is performed by the other endpoint. Once the `read` is performed, the `write` takes place.

The blocking and non-blocking `read` operations have analogous behaviors when
the pipe is empty.

### Defining a Pipe

SYCL*-compliant pipes are defined as a class with static members. To declare a pipe that
transfers integer data and has  `capacity=4`, use a type alias:

```c++
using ProducerToConsumerPipe = pipe<  // Defined in the SYCL headers.
  class ProducerConsumerPipe,         // An identifier for the pipe.
  int,                                // The type of data in the pipe.
  4>;                                 // The capacity of the pipe.
```

The `class ProducerToConsumerPipe` template parameter is important to the
uniqueness of the pipe. This class need not be defined but must be distinct
for each pipe. Consider another type alias with the same parameters:

```c++
using ProducerToConsumerPipe2 = pipe<  // Defined in the SYCL headers.
  class ProducerConsumerPipe,          // An identifier for the pipe.
  int,                                 // The type of data in the pipe.
  4>;                                  // The capacity of the pipe.
```

The uniqueness of a pipe is derived from a combination of all three template
parameters. Since `ProducerToConsumerPipe` and `ProducerToConsumerPipe2` have
the same template parameters, they define the same pipe.

### Using a Pipe

This code sample defines a `Consumer` and a `Producer` kernel connected
by the pipe `ProducerToConsumerPipe`. Kernels use the
`ProducerToConsumerPipe::write` and `ProducerToConsumerPipe::read` methods for
communication.

The `Producer` kernel reads integers from the global memory and writes those integers
into `ProducerToConsumerPipe`, as shown in the following code snippet:

```c++
void Producer(queue &q, buffer<int, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  auto e = q.submit([&](handler &h) {
    accessor input_accessor(input_buffer, h, read_only);
    auto num_elements = input_buffer.get_count();

    h.single_task<ProducerTutorial>([=]() {
      for (size_t i = 0; i < num_elements; ++i) {
        ProducerToConsumerPipe::write(input_accessor[i]);
      }
    });
  });
}
```

The `Consumer` kernel reads integers from `ProducerToConsumerPipe`, processes
the integers (`ConsumerWork(i)`), and writes the result into the global memory.

```c++
void Consumer(queue &q, buffer<int, 1> &output_buffer) {
  std::cout << "Enqueuing consumer...\n";

  auto e = q.submit([&](handler &h) {
    accessor out_accessor(out_buf, h, write_only, no_init);
    size_t num_elements = output_buffer.get_count();

    h.single_task<ConsumerTutorial>([=]() {
      for (size_t i = 0; i < num_elements; ++i) {
        int input = ProducerToConsumerPipe::read();
        int answer = ConsumerWork(input);
        output_accessor[i] = answer;
      }
    });
  });
}
```

> **Note**: The `read` and `write` operations used are blocking. If
`ConsumerWork` is an expensive operation, then `Producer` might fill
`ProducerToConsumerPipe` faster than `Consumer` can read from it, causing
`Producer` to block occasionally.

## Build the `Pipes` Tutorial

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
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
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
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

### Read the Reports

Locate `report.html` in the `pipes_report.prj/reports/` directory.

Navigate to the "System Viewer" to visualize the structure of the kernel system. Identify the pipe connecting the two kernels.

## Run the `Pipes` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./pipes.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./pipes.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./pipes.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   pipes.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   pipes.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   pipes.fpga.exe
   ```

## Example Output

**Example output when run on the FPGA emulator or simulator**

```
Input Array Size: 8192
Enqueuing producer...
Enqueuing consumer...

Profiling Info
  Producer:
    Start time: 0 ms
    End time: +8.18174 ms
    Kernel Duration: 8.18174 ms
  Consumer:
    Start time: +7.05307 ms
    End time: +8.18231 ms
    Kernel Duration: 1.12924 ms
  Design Duration: 8.18231 ms
  Design Throughput: 4.00474 MB/s

PASSED: The results are correct
```
> **Note**: The FPGA emulator or simulator does not accurately represent the performance nor the relative timing (the start and end times) for the kernel.

**Example output when run on an FPGA device**

```
Input Array Size: 1048576
Enqueuing producer...
Enqueuing consumer...

Profiling Info
  Producer:
    Start time: 0 ms
    End time: +4.481 ms
    Kernel Duration: 4.481 ms
  Consumer:
    Start time: +0.917 ms
    End time: +4.484 ms
    Kernel Duration: 3.568 ms
  Design Duration: 4.484 ms
  Design Throughput: 935.348 MB/s

PASSED: The results are correct
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).