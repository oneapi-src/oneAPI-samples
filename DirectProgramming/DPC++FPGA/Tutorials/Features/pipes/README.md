# Data Transfers Using Pipes
This FPGA tutorial shows how to use pipes to transfer data between kernels.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | The basics of the of DPC++ pipes extension for FPGA<br> How to declare and use pipes in a DPC++ program
| Time to complete                  | 15 minutes



## Purpose
This tutorial demonstrates how a kernel in a DPC++ FPGA program transfers
data to or from another kernel using the pipe abstraction.

### Definition of a Pipe
The primary goal of pipes is to allow concurrent execution of kernels that need
to exchange data.

A pipe is a FIFO data structure connecting two endpoints that communicate
using the pipe's `read` and `write` operations. An endpoint can be either a kernel
or an external I/O on the FPGA. Therefore, there are three types of pipes: 
* kernel-kernel
* kernel-I/O 
* I/O-kernel 

This tutorial focuses on kernel-kernel pipes, but
the concepts discussed here apply to other kinds of pipes as well.

The `read` and `write` operations have two variants: 
* Blocking variant: Blocking operations may not return immediately but are always successful.
* Non-blocking variant: Non-blocking operations take an extra boolean parameter
that is set to `true` if the operation happened successfully. 

Data flows in a single direction inside pipes. In other words, for a pipe `P`
and two kernels using `P`, one of the kernels is exclusively going to perform
`write` to `P` while the other kernel is exclusively going to perform `read` from
`P`. Bidirectional communication can be achieved using two pipes.

Each pipe has a configurable `capacity` parameter describing the number of `write`
operations that may be performed without any `read` operations being performed. For example,
consider a pipe `P` with capacity 3, and two kernels `K1` and `K2` using
`P`. Assume that `K1` performed the following sequence of operations:

 `write(1)`, `write(2)`, `write(3)`

In this situation, the pipe is full because three (the `capacity` of
`P`) `write` operations were performed without any `read` operation. In this
situation, a `read` must occur before any other `write` is allowed.

If a `write` is attempted to a full pipe, one of two behaviors occur:

  * If the operation is non-blocking, it returns immediately, and its
  boolean parameter is set to `false`. The `write` does not have any effect.
  * If the operation is blocking, it does not return until a `read` is
  performed by the other endpoint. Once the `read` is performed, the `write`
  takes place.

The blocking and non-blocking `read` operations have analogous behaviors when
the pipe is empty.

### Defining a Pipe in DPC++

In DPC++, pipes are defined as a class with static members. To declare a pipe that
transfers integer data and has  `capacity=4`, use a type alias:

```c++
using ProducerToConsumerPipe = pipe<  // Defined in the DPC++ headers.
  class ProducerConsumerPipe,         // An identifier for the pipe.
  int,                                // The type of data in the pipe.
  4>;                                 // The capacity of the pipe.
```

The `class ProducerToConsumerPipe` template parameter is important to the
uniqueness of the pipe. This class need not be defined but must be distinct
for each pipe. Consider another type alias with the exact same parameters:

```c++
using ProducerToConsumerPipe2 = pipe<  // Defined in the DPC++ headers.
  class ProducerConsumerPipe,          // An identifier for the pipe.
  int,                                 // The type of data in the pipe.
  4>;                                  // The capacity of the pipe.
```

The uniqueness of a pipe is derived from a combination of all three template
parameters. Since `ProducerToConsumerPipe` and `ProducerToConsumerPipe2` have
the same template parameters, they define the same pipe.

### Using a Pipe in DPC++

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
    accessor out_accessor(out_buf, h, write_only, noinit);
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

**NOTE:** The `read` and `write` operations used are blocking. If
`ConsumerWork` is an expensive operation, then `Producer` might fill
`ProducerToConsumerPipe` faster than `Consumer` can read from it, causing
`Producer` to block occasionally.

## Key Concepts
* The basics of the of DPC++ pipes extension for FPGA
* How to declare and use pipes in a DPC++ program

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `pipes` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/pipes.fpga.tar.gz" download>here</a>.

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

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report: 
     ```
     nmake report
     ``` 
   * An FPGA hardware target is not provided on Windows*. 

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.
 
 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `pipes_report.prj/reports/` or `pipes_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Navigate to the "System Viewer" to visualize the structure of the kernel system. Identify the pipe connecting the two kernels.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./pipes.fpga_emu     (Linux)
     pipes.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./pipes.fpga         (Linux)
     ```

### Example of Output
You should see the following output in the console:

1. When running on the FPGA emulator
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
    NOTE: The FPGA emulator does not accurately represent the performance nor the kernels' relative timing (i.e., the start and end times).

2. When running on the FPGA device
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
