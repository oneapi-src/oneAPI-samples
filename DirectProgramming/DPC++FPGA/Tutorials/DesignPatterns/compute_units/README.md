
# Using Compute Units To Duplicate Kernels
This FPGA tutorial showcases a design pattern that allows you to make multiple copies of a kernel, called compute units.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | A design pattern to generate multiple compute units in DPC++ using template metaprogramming
| Time to complete                  | 15 minutes



## Purpose
A design's performance can sometimes be increased by making multiple copies of kernels to make full use of the FPGA resources. Each kernel copy is called a compute unit.

This tutorial provides a header file that defines an abstraction for making multiple copies of a single-task kernel. This header can be used in any DPC++ design and can be extended as necessary.

### Compute units example

This code sample builds on the pipe_array tutorial. It includes the header file `unroller.hpp` from the pipe_array tutorial, and provides the `compute_units.hpp` header as well.

This code sample defines a `Source` kernel, and a `Sink` kernel. Data is read from host memory by the `Source` kernel and sent to the `Sink` kernel via a chain of compute units. The number of compute units in the chain between `Source` and `Sink` is determined by the following parameter:

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
    accessor out_accessor(out_buf, h, write_only, no_init);
    h.single_task<Sink>(
        [=] { out_accessor[0] = Pipes::PipeAt<kEngines>::read(); });
  });
}
```

 The following line defines each compute unit's functionality and submits each compute unit to the given queue `q`. The number of copies to submit is provided by the first template parameter of `SubmitComputeUnits`, while the second template parameter allows you to give the compute units a shared base name, e.g., `ChainComputeUnit`. The functionality of each compute unit must be specified by a generic lambda expression that takes a single argument (with type `auto`) that provides an ID for each compute unit:

``` c++
SubmitComputeUnits<kEngines, ChainComputeUnit>(q, [=](auto ID) {
    auto f = Pipes::PipeAt<ID>::read();
    Pipes::PipeAt<ID + 1>::write(f);
  });
```

Each compute unit in the chain from `Source` to `Sink` must read from a unique pipe and write to the next pipe. As seen above, each compute unit knows its ID and therefore, its behavior can depend on this ID. Each compute unit in the chain will read from pipe `ID` and write to pipe `ID + 1`.

## Key Concepts
* A design pattern to generate multiple compute units in DPC++ using template metaprogramming

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the `compute_units` Tutorial

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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/compute_units.fpga.tar.gz" download>here</a>.

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

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
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

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)


## Examining the Reports
Locate `report.html` in the `compute_units_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

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

