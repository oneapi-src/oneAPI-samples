
# Data Transfers Using Pipe Arrays
This FPGA tutorial showcases a design pattern that makes it possible to create arrays of pipes.

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. 

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | A design pattern to generate a array of pipes in DPC++ <br> Static loop unrolling through template metaprogramming
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose
In certain situations, it is useful to be able to create collection of pipes that can be indexed like an array in a DPC++ FPGA design. If you are not yet familiar with DPC++ pipes, refer to the prerequisite tutorial "Data Transfers Using Pipes".

In SYCL*, each pipe defines a unique type with static methods for reading data (`read`) and writing data (`write`). Since pipes are not objects but *types*, defining a collection of pipes requires C++ template meta-programming. This is somewhat non-intuitive but yields highly efficient code.

This tutorial provides a convenient pair of header files defining an abstraction for an array of pipes. The headers can be used in any DPC++ design and can be extended as necessary.

### Example 1: A simple array of pipes

To create an array of pipes, include the top-level header (from this code sample) in your design:

```c++
#include "pipe_array.hpp"
```

As with regular pipes, an array of pipes needs template parameters for an ID, for the `min_capacity` of each pipe, and for the data type of each pipe. An array of pipes additionally requires one or more template parameters to specify the array size. The following code declares a one dimensional array of 10 pipes, each with `capacity=32`, that operate on `int` values.

```c++
using MyPipeArray = PipeArray<     // Defined in "pipe_array.h".
    class MyPipe,                  // An identifier for the pipe.
    int,                           // The type of data in the pipe.
    32,                            // The capacity of each pipe.
    10,                            // array dimension.
    >;
```

The uniqueness of a pipe array is derived from a combination of all template parameters.

Indexing inside a pipe array can be done via the `PipeArray::PipeAt` type alias, as shown in the following code snippet:

```c++
MyPipeArray::PipeAt<3>::write(17);
auto x = MyPipeArray::PipeAt<3>::read();
```
The template parameter `<3>` identifies a specific pipe within the array of pipes.  The index of the pipe being accessed *must* be determinable at compile time. 

In most cases, we want to use an array of pipes so that we can iterate over them in a loop. In order to respect the requirement that all pipe indices are uniquely determinable at compile time, we must use a static form of loop unrolling based on C++ templates. A simple example is shown in the code snippet:

```c++
// Write 17 to every pipe in the array
Unroller<0, 10>::Step([](auto i) {
  MyPipeArray::PipeAt<i>::write(17);
});
```
While this may initially feel foreign to those unaccustomed to C++ template metaprogramming, this is a simple and powerful pattern common to many C++ libraries. It is easy to reuse. In addition to `pipe_array.hpp`, this code sample includes a simple header file `unroller.hpp`, which implements the  `Unroller` functionality.

### Example 2: A 2D array of pipes

This code sample defines a `Producer` kernel that reads data from host memory and forwards this data into a two dimensional pipe matrix. 

The following code snippet creates a two dimensional pipe array.
``` c++
constexpr size_t kNumRows = 2;
constexpr size_t kNumCols = 2;
constexpr size_t kDepth = 2;

using ProducerToConsumerPipeMatrix = PipeArray<  // Defined in "pipe_array.h".
    class ProducerConsumerPipe,                  // An identifier for the pipe.
    uint64_t,                                    // The type of data in the pipe.
    kDepth,                                      // The capacity of each pipe.
    kNumRows,                                    // array dimension.
    kNumCols                                     // array dimension.
    >;
```
The producer kernel writes `num_passes` units of data into each of the `kNumRows * kNumCols` pipes. Note that the unrollers' lambdas must capture certain variables from their outer scope.

```c++
h.single_task<ProducerTutorial>([=]() {
  size_t input_idx = 0;
  for (size_t pass = 0; pass < num_passes; pass++) {
    // Template-based unroll (outer "i" loop)
    Unroller<0, kNumRows>::Step([&input_idx, input_accessor](auto i) {
      // Template-based unroll (inner "j" loop)
      Unroller<0, kNumCols>::Step([&input_idx, i, input_accessor](auto j) {
        // Write a value to the <i,j> pipe of the pipe array
        ProducerToConsumerPipeMatrix::PipeAt<i, j>::write(
            input_accessor[input_idx++]);
      });
    });
  }
});
```

The code sample also defines an array of `Consumer` kernels that each read from a unique pipe in `ProducerToConsumerPipeMatrix`, process the data, and write the result to the host memory. 

```c++
// The consumer kernel reads from a single pipe, determined by consumer_id
h.single_task<ConsumerTutorial<consumer_id>>([=]() {
  constexpr size_t x = consumer_id / kNumCols;
  constexpr size_t y = consumer_id % kNumCols;
  for (size_t i = 0; i < num_elements; ++i) {
    auto input = ProducerToConsumerPipeMatrix::PipeAt<x,y>::read();
    uint64_t answer = ConsumerWork(input); // do some processing
    output_accessor[i] = answer;
  }
});
```

The host must thus enqueue the producer kernel and `kNumRows * kNumCols` separate consumer kernels. The latter is achieved through another static unroll.
```c++
{
  queue q(device_selector, dpc_common::exception_handler);

  // Enqueue producer
  buffer<uint64_t,1> producer_buffer(producer_input);
  Producer(q, producer_buffer);
  
  // Use template-based unroll to enqueue multiple consumers    
  std::vector<buffer<uint64_t,1>> consumer_buffers;
  Unroller<0, kNumberOfConsumers>::Step([&](auto consumer_id) {
    consumer_buffers.emplace_back(consumer_output[consumer_id].data(), items_per_consumer);
    Consumer<consumer_id>(q, consumer_buffers.back());
  });
}
```

## Key Concepts
* A design pattern to generate a array of pipes in DPC++
* Static loop unrolling through template metaprogramming

## License  
This code sample is licensed under MIT license.


## Building the `pipe_array` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile or fpga_runtime) as well as whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/get-started/base-toolkit/](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)).

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
   Alternatively, to compile for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:

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
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/pipe_array.fpga.tar.gz" download>here</a>.

### On a Windows* System
Note: `cmake` is not yet supported on Windows. A build.ninja file is provided instead. 

1. Enter the source file directory.
   ```
   cd src
   ```

2. Compile the design. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
      ```
      ninja fpga_emu
      ```
      **NOTE:** For the FPGA emulator target, the device link method is used. 
   * Generate the optimization report:

     ```
     ninja report
     ```
     If you are targeting Intel® PAC with Intel Stratix® 10 SX FPGA, instead use:
     ```
     ninja report_s10_pac
     ```     
   * Compiling for FPGA hardware is not yet supported on Windows.
 
 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)


## Examining the Reports
Locate `report.html` in the `pipe_array_report.prj/reports/` or `pipe_array_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

You can visualize the kernels and pipes generated by looking at the "System Viewer" section of the report. However, it is recommended that you first reduce the array dimensions `kNumRows` and `kNumCols` to small values (2 or 3) to facilitate visualization.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./pipe_array.fpga_emu     (Linux)
     pipe_array.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./pipe_array.fpga         (Linux)
     ```

### Example of Output
```
Input Array Size:  1024
Enqueuing producer...
Enqueuing consumer 0...
Enqueuing consumer 1...
Enqueuing consumer 2...
Enqueuing consumer 3...
PASSED: The results are correct
```
