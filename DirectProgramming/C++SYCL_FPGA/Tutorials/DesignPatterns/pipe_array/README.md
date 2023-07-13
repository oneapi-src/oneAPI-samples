
# "Pipe Array" Sample

This FPGA sample demonstrates a design pattern to create arrays of pipes.

| Area                 | Description
|:--                   |:--
| What you will learn  | A design pattern to generate an array of pipes using SYCL* <br> Static loop unrolling through template metaprogramming
| Time to complete     | 15 minutes
| Category             | Code Optimization

## Purpose

In certain situations, it is useful to create a collection of pipes that can be indexed like an array in a SYCL-compliant FPGA design. If you are not yet familiar with pipes, refer to the prerequisite tutorial "Data Transfers Using Pipes".

In SYCL*, each pipe defines a unique type with static methods for reading data (`read`) and writing data (`write`). Since pipes are not objects but *types*, defining a collection of pipes requires C++ template meta-programming. This is somewhat non-intuitive but yields highly efficient code.

This tutorial provides a convenient pair of header files defining an abstraction for an array of pipes. The headers can be used in any SYCL-compliant design and can be extended as necessary.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel® oneAPI DPC++/C++ Compiler is enough to compile for emulation, generating reports, generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, you must have Intel® Quartus® Prime Pro Edition and one of the following simulators installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstrates a design pattern.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.

## Key Implementation Details

The sample demonstrates the following important concepts:

- A design pattern to generate an array of pipes.
- Static loop unrolling through template metaprogramming.

### A Simple Array of Pipes

To create an array of pipes, include the `pipe_utils.hpp` header from the `../DirectProgramming/C++SYCL_FPGA/include/..` directory in your design:

```c++
#include "pipe_utils.hpp"
```

As with regular pipes, an array of pipes needs template parameters for an ID, for the `min_capacity` of each pipe, and each pipe's data type. An array of pipes additionally requires one or more template parameters to specify the array size. The following code declares a one-dimensional array of 10 pipes, each with `capacity=32`, that operate on `int` values.

```c++
using MyPipeArray = PipeArray<     // Defined in "pipe_utils.hpp".
    class MyPipe,                  // An identifier for the pipe.
    int,                           // The type of data in the pipe.
    32,                            // The capacity of each pipe.
    10                             // array dimension.
    >;
```

The uniqueness of a pipe array is derived from a combination of all template parameters.

Indexing inside a pipe array can be done via the `PipeArray::PipeAt` type alias, as shown in the following code snippet:

```c++
MyPipeArray::PipeAt<3>::write(17);
auto x = MyPipeArray::PipeAt<3>::read();
```

The template parameter `<3>` identifies a specific pipe within the array of pipes.  The index of the pipe being accessed *must* be determinable at compile time.

In most cases, we want to use an array of pipes so that we can iterate over them in a loop. To respect the requirement that all pipe indices are uniquely determinable at compile time, we must use a static form of loop unrolling based on C++ templates. A simple example is shown in the code snippet:

```c++
// Write 17 to every pipe in the array
Unroller<0, 10>::Step([](auto i) {
  MyPipeArray::PipeAt<i>::write(17);
});
```

While this approach may feel foreign to those unaccustomed to C++ template metaprogramming, this is a simple and powerful pattern common to many C++ libraries. It is easy to reuse. This code sample includes a simple header file `unroller.hpp`, which implements the  `Unroller` functionality.

### A 2D Array of Pipes

This code sample defines a `Producer` kernel that reads data from host memory and forwards this data into a two-dimensional pipe matrix. The following code snippet creates a two-dimensional pipe array.

``` c++
constexpr size_t kNumRows = 2;
constexpr size_t kNumCols = 2;
constexpr size_t kDepth = 2;

using ProducerToConsumerPipeMatrix = PipeArray<  // Defined in "pipe_utils.hpp".
    class ProducerConsumerPipe,                  // An identifier for the pipe.
    uint64_t,                                    // The type of data in the pipe.
    kDepth,                                      // The capacity of each pipe.
    kNumRows,                                    // array dimension.
    kNumCols                                     // array dimension.
    >;
```

The producer kernel writes `num_passes` units of data into each of the `kNumRows * kNumCols` pipes. Note that the lambdas in the unrollers must capture certain variables from their outer scope.

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
  queue q(device_selector, fpga_tools::exception_handler);

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

## Build the `Pipe Array` Sample

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
> For more information on configuring environment variables, see [*Use the setvars Script with Linux* or macOS**](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [*Use the setvars Script with Windows**](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

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

    1. Compile for emulation (fast compile time, targets emulated FPGA device):
       ```
       make fpga_emu
       ```
    2. Generate the optimization report:
       ```
       make report
       ```
       The report resides at `pipe_array_report.prj/reports/report.html`.

       You can visualize the kernels and pipes generated by looking at the *System Viewer* section of the report. However, you should first reduce the array dimensions `kNumRows` and `kNumCols` to small values (2 or 3) to help visualization.

    3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
       ```
       make fpga_sim
       ```
    4. Compile for FPGA hardware (longer compile time, targets FPGA device):
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

   1. Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report:
      ```
      nmake report
      ```
      The report resides at `pipe_array_report.prj.a/reports/report.html`.

      You can visualize the kernels and pipes generated by looking at the *System Viewer* section of the report. However, you should first reduce the array dimensions `kNumRows` and `kNumCols` to small values (2 or 3) to help visualization.

   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Pipe Array` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./pipe_array.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./pipe_array.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./pipe_array.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   pipe_array.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   pipe_array.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   pipe_array.fpga.exe
   ```

## Example Output

```
Input Array Size:  1024
Enqueuing producer...
Enqueuing consumer 0...
Enqueuing consumer 1...
Enqueuing consumer 2...
Enqueuing consumer 3...
PASSED: The results are correct
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
