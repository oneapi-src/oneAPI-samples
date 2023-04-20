
# `Compute Units` Sample

This `Compute Units` sample is a tutorial that showcases a design pattern to allow you to make multiple copies of a kernel, called compute units. Using Compute Units To Duplicate Kernels

| Area                 | Description
|:---                  |:---
| What you will learn  | How to use this design pattern to create of duplicate compute units
| Time to complete     | 15 minutes
| Category             | Code Optimization

## Purpose

The performance of a design can be increased sometimes by making multiple copies of kernels to make full use of the FPGA resources. Each kernel copy is called a compute unit.

This tutorial provides a header file that defines an abstraction for making multiple copies of a single-task kernel. This header can be used in any SYCL*-compliant design and can be extended as necessary.

## Prerequisites

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware           | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software           | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Key Implementation Details

The code in this sample is a design pattern to generate multiple compute units using SYCL-compliant template metaprogramming.

### Compute Units Concepts

The code included in the sample builds on the pipe_array tutorial. This sample includes the header file `unroller.hpp` from the pipe_array tutorial, and provides the `compute_units.hpp` header as well.

This sample defines a `Source` kernel and a `Sink` kernel. Data is read from host memory by the `Source` kernel and sent to the `Sink` kernel through a chain of compute units. The number of compute units in the chain between `Source` and `Sink` is determined by the following parameter:

``` c++
constexpr size_t kEngines = 5;
```

Pipes pass data between kernels. The number of pipes needed is equal to the length of the chain: the number of compute units, plus one. This array of pipes is declared as follows:

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

The following line defines functionality for each compute unit and submits each compute unit to the given queue `q`. The number of copies to submit is provided by the first template parameter of `SubmitComputeUnits`, while the second template parameter allows you to give the compute units a shared base name, e.g., `ChainComputeUnit`. The functionality of each compute unit must be specified by a generic lambda expression that takes a single argument (with type `auto`) that provides an ID for each compute unit:

``` c++
SubmitComputeUnits<kEngines, ChainComputeUnit>(q, [=](auto ID) {
    auto f = Pipes::PipeAt<ID>::read();
    Pipes::PipeAt<ID + 1>::write(f);
  });
```

Each compute unit in the chain from `Source` to `Sink` must read from a unique pipe and write to the next pipe. As seen above, each compute unit knows its ID; therefore, its behavior can depend on this ID. Each compute unit in the chain will read from pipe `ID` and write to pipe `ID + 1`.

## Build the `Compute Units` Sample

> **Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables.
> Set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation every time you open a new terminal window.
> This practice ensures that your compiler, libraries, and tools are ready for development.
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
2. Build the program for the Agilex® 7 device family, which is the default.

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

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      make fpga_sim
      ```
   3. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `compute_units_report.prj/reports/report.html`. You can visualize the kernels and pipes generated by looking at the "System Viewer" section of the report. Note that each compute unit is shown as a unique kernel in the reports, with names `ChainComputeUnit<0>`, `ChainComputeUnit<1>`, and so on.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Agilex® 7 device family, which is the default.
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
   2. Compile for simulation (fast compile time, targets simulator FPGA device):
      ```
      nmake fpga_sim
      ```
   3. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `compute_units_report.prj.a/reports/report.html`. You can visualize the kernels and pipes generated by looking at the "System Viewer" section of the report. Note that each compute unit is shown as a unique kernel in the reports, with names `ChainComputeUnit<0>`, `ChainComputeUnit<1>`, and so on.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your `build` directory in a shorter path, for example `C:\samples\build`. You can then build the sample in the new location, but you must specify the full path to the build files.

## Run the `Compute Units` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernels execute on the CPU).
   ```
   ./compute_units.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./compute_units.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./compute_units.fpga
   ```
### On Windows

1. Run the sample on the FPGA emulator (the kernels execute on the CPU).
   ```
   compute_units.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   compute_units.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   compute_units.fpga.exe
   ```

## Example Output
```
PASSED: The results are correct
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
