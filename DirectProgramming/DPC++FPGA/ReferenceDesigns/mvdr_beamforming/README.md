# `MVDR Beamforming` Sample

This reference design demonstrates IO streaming using SYCL* on an FPGA for a large system. The IO streaming is 'faked' using data from the host.

| Area                 | Description
|:---                  |:---
| What you will learn  | How to create a full, complex system that performs IO streaming using SYCL*-compliant code.
| Time to complete     | 1 hour
| Category             | Reference Designs and End to End

## Purpose

The purpose of this reference design is to implement a high-performance streaming IO design. In this reference design, we implement an MVDR-beamforming algorithm using oneAPI.

This reference design code sample leverages concepts that are discussed in the following FPGA tutorials:

- **IO Streaming** (io_streaming)
- **Explicit Pipelining with `fpga_reg`** (fpga_reg)
- **Loop `ivdep` Attribute** (loop_ivdep)
- **Triangular Loop Optimization** (triangular_loop)
- **Unrolling Loops** (loop_unroll)
- **Pipe Arrays** (pipe_array)

You should review the **IO Streaming** code sample as this reference design is a direct extension of the concepts described in that tutorial. The IO Streaming sample code clearly illustrates the concept of 'fake' IO Pipes, which is used heavily in this reference design.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA (Intel® PAC with Intel® Arria® 10 GX FPGA) <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX)
| Software             | Intel® oneAPI DPC++/C++ Compiler <br> Intel® FPGA Add-on for oneAPI Base Toolkit

## Key Implementation Details

### MVDR Beamforming

>**Note**: This reference design is built upon the **IO Streaming** sample code and concepts. Review that tutorial for more information.

The images below show the data flow in the MVDR beamforming design. The first image shows the "real" data flow when IO pipes are used at the inputs and outputs. The second image shows the data flow in this reference design where we don't have access to a BSP with IO pipes.

![processing_kernels_ideal](processing_kernels_ideal.png)

![processing_kernels_fake](processing_kernels_fake.png)

The `DataProducer` kernel replaces the input IO pipe in the first image. The splitting of data between the training and beamforming pipelines is done by the `InputDemux` kernel. The `DataOutConsumer` kernel replaces the output IO pipe in the first image. The data for the `SteeringVectorGenerator` kernel still comes from the host through the `SinThetaProducer` kernel. This kernel does not replace an IO pipe but simplifies and modularizes the host's data streaming to the device.

### Source Code

| File                       | Description
|:---                        |:---
|`mvdr_beamforming.cpp`      | Contains the `main()` function and the top-level interfaces to the MVDR functions
|`BackwardSubstitution.hpp`  | Backward Substitution kernel
|`Beamformer.hpp`            | Beamformer kernel, multiplies input vectors by each weight vector to generate final output
|`CalcWeights.hpp`           | CalcWeights kernel, multiplies BackwardSubstitution output by steering vectors
|`Constants.hpp`             | Defines constants used throughout the design, some can be overridden from the command line during compilation
|`FakeIOPipes.hpp`           | Implements 'fake' IO pipes, which interface to the host
|`ForwardSubstitution.hpp`   | Forward Substitution kernel
|`InputDemux.hpp`            | InputDemux kernel, separates training and processing data
|`mvdr_complex.hpp`          | Definition of ComplexType, used throughout this design
|`MVDR.hpp`                  | Function to launch all MVDR kernels and define the pipes that connect them together
|`ParallelCopyArray.hpp`     | Defines the ParallelCopyArray class, an array that supports unrolled copy / assign operations
|`pipe_utils.hpp`            | Header file containing the definition of an array of pipes and a pipe duplicator. This header can be found in the ../include/ directory of this repository.
|`SteeringVectorGenerator.hpp`   | SteeringVectorGenerator kernel, generates steering vectors based on data from the host
|`StreamingQRD.hpp`          | StreamingQRD kernel, performs Q-R Decomposition on a matrix
|`Transpose.hpp`             | Transpose kernel, reorders data for the StreamingQRD kernel
|`Tuple.hpp`                 | A templated tuple that defines the NTuple class which is used for pipe interfaces
|`udp_loopback_test.cpp`     | Contains the `main()` function for the loopback test. This code is only relevant for use with real IO pipes
|`UDP.hpp`                   | This code is **only** relevant for using the real IO pipes (for example not in Intel® DevCloud). This is discussed later in the [Using Real IO-pipes Section](#using-real-io-pipes)
|`UnrolledLoop.hpp`          | A templated-based loop unroller that unrolls loops in the compiler front end

### Additional Documentation

- [Explore SYCL* Through Intel® FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel® oneAPI Toolkits.
- [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel® oneAPI Toolkits.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `MVDR Beamforming` Design

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
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the [Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to
this readme for instructions on how to build and run a sample.

### On Linux*

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
       ```
       make fpga_emu
       ```
   2. Generate the HTML performance report.
       ```
       make report
       ```
      The report resides at `mvdr_beamforming_report.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
       ```
       make fpga
       ```

   (Optional) The hardware compiles listed above can take several hours to complete; alternatively, you can download FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) from [https://iotdk.intel.com/fpga-precompiled-binaries/latest/mvdr_beamforming.fpga.tar.gz](https://iotdk.intel.com/fpga-precompiled-binaries/latest/mvdr_beamforming.fpga.tar.gz).

### On Windows*

> **Note**: The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.

1. Change to the sample directory.
2. Build the program for **Intel® PAC with Intel Arria® 10 GX FPGA**, which is the default
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   For the **Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX)**, enter the following:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=intel_s10sx_pac:pac_s10
   ```
3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the HTML performance report.
      ```
      nmake report
      ```
      The report resides at `mvdr_beamforming_report.a.prj/reports/report.html`.

   3. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `MVDR Beamforming` Design

### Configurable Parameters

The general syntax for running the program is shown below and the table describes the index values.

`<program> <Index 0> <Index 1> <Index 2>`

| Argument Index | Description
|:---            |:---
| 0              | The number of matrices (default=`1024`)
| 1              | The input directory (default=`../data`)
| 2              | The output directory (default=`.`)

### On Linux
 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
    ```
    ./mvdr_beamforming.fpga_emu 1024 ../data .
    ```
2. Run the sample on the FPGA device.
   ```
   ./mvdr_beamforming.fpga 1024 ../data .
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   mvdr_beamforming.fpga_emu.exe 1024 ../data .
   ```
2. Run the sample on the FPGA device.
   ```
   mvdr_beamforming.fpga.exe 1024 ../data .
   ```

### Build and Run the Sample on Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

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

Only `fpga_compile` nodes support compiling to FPGA. When compiling for FPGA hardware, increase the job timeout to **24 hours**.

Executing programs on FPGA hardware is only supported on `fpga_runtime` nodes of the appropriate type, such as `fpga_runtime:arria10` or `fpga_runtime:stratix10`.

Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® DevCloud for oneAPI [*Intel® oneAPI Base Toolkit Get Started*](https://devcloud.intel.com/oneapi/get_started/) page.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.


## Build and Run the Design Using Real IO-pipes

This section describes how to build and run this reference design on a BSP with real IO pipes. The real IO pipes version does **not** work on Windows and requires a specific system setup and BSP.

>**Note**: This design requires a specific board support package (BSP) with a distinct hardware configuration. For access to this BSP or general customer support, submit a case through Intel® Premier Support (IPS) or contact your Intel or Distribution Sales Representative.

### Build on Linux

1. Build the loopback test and reference design with real IO pipes.
   ```
   mkdir build
   cd build
   cmake .. -DREAL_IO_PIPES=1 -DFPGA_DEVICE=pac_s10_usm_udp
   ```
   The `REAL_IO_PIPES` cmake flag defines a variable that is used *exclusively* in `mvdr_beamforming.cpp` to create a kernel system using real IO pipes, as opposed to the fake IO pipes described earlier in this document.

2. Build the loopback test only.
   ```
   make udp_loopback_test
   ```
3. Build the MVDR reference design only.
   ```
   make fpga
   ```
### Run on Linux

1. Run the loopback test and reference design with real IO pipes.
   ```
   ./udp_loopback_test.fpga 64:4C:36:00:2F:20 192.168.0.11 34543 255.255.255.0 94:40:C9:71:8D:10 192.168.0.10 34543 10000000
   ```

   The general syntax for running the program is shown below and the table describes the index values.

   `<program> <Index 1> <Index 2> <Index 3> <Index 4> <Index 5> <Index 6> <Index 7> <Index 8>`

   | Argument Index | Description
   |:---            |:---
   | 1              | FPGA Media Access Control (MAC) Address
   | 2              | FPGA Internet Protocol (IP) Address
   | 3              | FPGA User Datagram Protocol (UDP) Port
   | 4              | FPGA Netmask
   | 5              | Host Media Access Control (MAC) Address
   | 6              | Host Internet Protocol (IP) Address
   | 7              | Host User Datagram Protocol (UDP) Port
   | 8              | Number of packets (optional, default=`100000000`)

2. Run the MVDR reference design with real IO pipes.
   ```
   ./mvdr_beamforming.fpga 64:4C:36:00:2F:20 192.168.0.11 34543 255.255.255.0 94:40:C9:71:8D:10 192.168.0.10 34543 1024 ../data .
   ```
   The general syntax for running the program is shown below and the table describes the index values.

   `<program> <Index 1> <Index 2> <Index 3> <Index 4> <Index 5> <Index 6> <Index 7> <Index 8> <Index 9> <Index 10>`

   | Argument Index | Description
   |:---            |:---
   | 1              | FPGA Media Access Control (MAC) Address
   | 2              | FPGA Internet Protocol (IP) Address
   | 3              | FPGA User Datagram Protocol (UDP) Port
   | 4              | FPGA Netmask
   | 5              | Host Media Access Control (MAC) Address
   | 6              | Host Internet Protocol (IP) Address
   | 7              | Host User Datagram Protocol (UDP) Port
   | 8              | The number of matrices (optional, default=`1024`)
   | 9              | The input directory (optional, default=`../data`)
   | 10             | The output directory (optional, default=`.`


## Example Output

```
Matrices:         1024
Input Directory:  '../data'
Output Directory: '.'

Reading training data from '../data/A_real.txt and ../data/A_imag.txt
Reading input data from ../data/X_real.txt and ../data/X_imag.txt
Launched MVDR kernels

*** Launching throughput test of 1024 matrices ***
Sensor inputs                 : 16
Training matrix rows          : 48
Data rows per training matrix : 48
Steering vectors              : 25
Streaming pipe width          : 4
Throughput: 233793 matrices/second
Checking output data against ../data/small_expected_out_real.txt and ../data/small_expected_out_imag.txt
Output data check succeeded
PASSED
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).