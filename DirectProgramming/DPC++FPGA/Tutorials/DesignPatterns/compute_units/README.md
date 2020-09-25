
# Using Compute Units To Duplicate Kernels
This FPGA tutorial showcases a design pattern that allows you to make multiple copies of a kernel, called compute units.

***Documentation***: The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming. 

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | A design pattern to generate multiple compute units in DPC++ using template metaprogramming
| Time to complete                  | 15 minutes

_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose
The performance of a design can sometimes by increased by making multiple copies of kernels, to make full use of the FPGA resources. Each kernel copy is called a compute unit.  

This tutorial provides a header file that defines an abstraction for making multiple copies of a single-task kernel. This header can be used in any DPC++ design and can be extended as necessary.

### Compute units example 

This code sample builds on the pipe_array tutorial. It includes the header files `pipe_array.hpp`, `pipe_array_internal.hpp`, and `unroller.hpp` from the pipe_array tutorial, and provides the `compute_units.hpp` header as well.

This code sample defines a `Source` kernel, and a `Sink` kernel. Data is read from host memory by the `Source` kernel, and sent to the `Sink` kernel via a chain of compute units. The number of compute units in the chain between `Source` and `Sink` is determined by the following parameter:

``` c++
constexpr size_t kEngines = 5;
```

Data is passed between kernels using pipes. The number of pipes needed is equal to the length of the chain, i.e., the number of compute units, plus one.
This array of pipes is declared as follows:

``` c++
using Pipes = PipeArray<class MyPipe, float, 1, kEngines + 1>;
```

The `Source` kernel reads data from host memory and writes it to the first pipe:

``` c++
void SourceKernel(queue &q, float data) {

  q.submit([&](handler &h) {
    h.single_task<Source>([=] { Pipes::PipeAt<0>::write(data); });
  });
}
```

At the end of the chain, the `Sink` kernel reads data from the last pipe and returns it to the host:

``` c++
void SinkKernel(queue &q, float &out_data) {

  // The verbose buffer syntax is necessary here,
  // since out_data is just a single scalar value
  // and its size can not be inferred automatically
  buffer<float, 1> out_buf(&out_data, 1);

  q.submit([&](handler &h) {
    auto out_accessor = out_buf.get_access<access::mode::write>(h);
    h.single_task<Sink>(
        [=] { out_accessor[0] = Pipes::PipeAt<kEngines>::read(); });
  });
}
```

 The following line defines the functionality of each compute unit, and submits each compute unit to the given queue `q`. The number of copies to submit is provided by the first template parameter of `SubmitComputeUnits`, while the second template parameter allows you to give the compute units a shared base name, e.g., `ChainComputeUnit`. The functionality of each compute unit must be specified by a generic lambda expression, that takes a single argument (with type `auto`) that provides an ID for each compute unit:

``` c++
SubmitComputeUnits<kEngines, ChainComputeUnit>(q, [=](auto ID) {
    auto f = Pipes::PipeAt<ID>::read();
    Pipes::PipeAt<ID + 1>::write(f);
  });
```

Each compute unit in the chain from `Source` to `Sink` must read from a unique pipe, and write to the next pipe. As seen above, each compute unit knows its own ID and therefore its behavior can depend on this ID. Each compute unit in the chain will read from pipe `ID` and write to pipe `ID + 1`.

## Key Concepts
* A design pattern to generate multiple compute units in DPC++ using template metaprogramming

## License  
This code sample is licensed under MIT license.


## Building the `compute_units` Tutorial

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
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/compute_units.fpga.tar.gz" download>here</a>.

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
Locate `report.html` in the `compute_units_report.prj/reports/` or `compute_units_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

You can visualize the kernels and pipes generated by looking at the "System Viewer" section of the report. Note that each compute unit is shown as a unique kernel in the reports, with names `ChainComputeUnit<0>`, `ChainComputeUnit<1>`, etc.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernels execute on the CPU):
     ```
     ./compute_units.fpga_emu     (Linux)
     compute_units.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./compute_units.fpga         (Linux)
     ```

### Example of Output
```
PASSED: The results are correct
```
