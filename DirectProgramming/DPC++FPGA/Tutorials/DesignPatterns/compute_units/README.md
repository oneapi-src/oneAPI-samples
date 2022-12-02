
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

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware           | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> FPGA third-party/custom platforms with oneAPI support
| Software           | Intel® oneAPI DPC++/C++ Compiler

> Note: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software required for the simulation flow and FPGA compiles.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> Questa*-Intel® FPGA Edition
> Questa*-Intel® FPGA Starter Edition
> ModelSim® SE
>
> For using the hardware compile flow, Intel Quartus Prime Pro Edition must be installed and accessible through your PATH.

>**Note**: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*.

### Additional Documentation

- *[Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html)* helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- *[FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)* helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)* helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

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

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Compute Units` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
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
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the *[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```
   For **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   For a custom FPGA platform, ensure that the board support package is installed on your system then enter a command similar to the following:

   ```
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      make report
      ```
      The report resides at `compute_units_report.prj/reports/report.html`. You can visualize the kernels and pipes generated by looking at the "System Viewer" section of the report. Note that each compute unit is shown as a unique kernel in the reports, with names `ChainComputeUnit<0>`, `ChainComputeUnit<1>`, and so on.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      make fpga
      ```
   (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/compute_units.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/compute_units.fpga.tar.gz).


### On Windows*

>**Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   To compile for the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
   For a custom FPGA platform, ensure that the board support package is installed on your system then enter a command similar to the following:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate HTML performance report.
      ```
      nmake report
      ```
      The report resides at `compute_units_report.prj.a/reports/report.html`. You can visualize the kernels and pipes generated by looking at the "System Viewer" section of the report. Note that each compute unit is shown as a unique kernel in the reports, with names `ChainComputeUnit<0>`, `ChainComputeUnit<1>`, and so on.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your `build` directory in a shorter path, for example `C:\samples\build`. You can then build the sample in the new location, but you must specify the full path to the build files.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `Compute Units` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernels execute on the CPU).
   ```
   ./compute_units.fpga_emu
   ```
2. Run the sample on the FPGA device.
   ```
   ./compute_units.fpga
   ```
### On Windows

1. Run the sample on the FPGA emulator (the kernels execute on the CPU).
   ```
   compute_units.fpga_emu.exe
   ```
2. Run the sample on the FPGA device.
   ```
   compute_units.fpga.exe
   ```

### Build and Run the Samples on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.


Use the Linux instructions to build and run the program.

You can specify an FPGA runtime node using a single line script similar to the following example.

```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:fpga_runtime:ppn=2` (lower case L) assigns one full node.
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes           |Command Options
  |:---                      |:---
  |FPGA Compile Time         |`qsub -l nodes=1:fpga_compile:ppn=2 -d .`
  |FPGA Runtime (Arria 10)   |`qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d .`
  |FPGA Runtime (Stratix 10) |`qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d .`
  |GPU	                    |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU	                    |`qsub -l nodes=1:xeon:ppn=2 -d .`

>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **12 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.

## Example Output
```
PASSED: The results are correct
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
