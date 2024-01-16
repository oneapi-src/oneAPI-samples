# `annotated_ptr` Sample

This tutorial demonstrates how to use `annotated_ptr` to annotate properties to pointers inside the kernel, which enables certain compiler optimization that reduces the area usage of the FPGA IP components produced by the Intel® oneAPI DPC++/C++ Compiler.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware                          | Intel® Agilex® 7, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | Best practices for creating and managing a oneAPI FPGA project
| Time to complete                  | 15 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> To use the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 2 sample that helps you getting started.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier2 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), etc.

## Purpose

The `hls_flow_interface/mmhost` sample (/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/hls_flow_interfaces/mmhost) demonstrates the usage of `annotated_arg` class in customizing Avalon memory-mapped interfaces for the FPGA IP component.
Sometimes annotations need to be applied on pointers inside the kernel to enable certain compiler optimizations. This tutorial shows how to use `annotated_ptr` to annotate a specific buffer location to a pointer variable inside the kernel, which reduces the LSUs use in the produced FPGA IP component.

### Using `annotated_ptr` to annotate the buffer location of a pointer
In the example, the device code defines a SYCL kernel functor that computes the dot product between a weight matrix (located in buffer location 1) and a vector (located in buffer location 2), and saves to the result vector located in buffer location 1.

Both of the input and output vectors are kernel arguments annotated by `annotated_arg` class:
```
struct pipeWithAnnotatedPtr {
  annotated_arg<float *, decltype(properties{buffer_location<kBL2>})> in_vec;
  annotated_arg<float *, decltype(properties{buffer_location<kBL1>})> out_vec;
  ...
};
```

The address of the weight matrix is transferred into the kernel via a host pipe. The kernel reads the row pointers of the weight matrix:
```
using MyPipe = ext::intel::experimental::pipe<class MyPipeName, float *>;
...
float *p = MyPipe::read();
```

Since the pointer `p` is read from a pipe, the buffer location of the pointer is not inferrable. The compiler will build two sets of LSUs that connect to buffer location 1 and buffer location 2 respectively, which increases the kernel area.

To resolve this, you can provide the buffer location information to the compiler by using `annotated_ptr` as follows
```
annotated_ptr<float, decltype(properties{buffer_location<1>})> mat{p};
```
As a result, the compiler will build only one set of LSUs that connect to buffer location 1.

The `annotated_ptr` variable `mat` is then used in the dot-product computation
```
float sum = 0.0f;
#pragma unroll COLS
for (int j = 0; j < COLS; j++)
   sum += mat[j] * in_vec[j];
out_vec[i] = sum;
```

## Building the `annotated_ptr` Tutorial
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

Use these commands to run the design, depending on your OS.

### On a Linux* System
This design uses CMake to generate a build script for GNU/make.

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake ..
   ```

3. Compile the design with the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

	   | Compilation Type    | Command
	   |:---                 |:---
	   | FPGA Emulator       | `make fpga_emu`
	   | Optimization Report | `make report`
	   | FPGA Simulator      | `make fpga_sim`

### On a Windows* System
This design uses CMake to generate a build script for  `nmake`.

1. Change to the sample directory.

2. Configure the build system for the Agilex® 7 device family, which is the default.

   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   
3. Compile the design with the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

	   | Compilation Type    | Command (Windows)
	   |:---                 |:---
	   | FPGA Emulator       | `nmake fpga_emu`
	   | Optimization Report | `nmake report`
	   | FPGA Simulator      | `nmake fpga_sim`
	   | FPGA Hardware       | `nmake fpga`


### Read the Reports

Build the `report` target and locate `report.html` in the `annotated_ptr.report.prj/reports/` directory.

Navigate to *System Viewer* (*Views* > *System Viewer*) and click on *Global memory* in the *System* hierarchy. Observe that the compiler generates a number of `COLS` LD nodes connected to global memory 1, which correspond to `COLS` times of read access over `data[j]` in the inner loop of the computation. You can verify that using the unannotated pointer `p[j]` directly in the inner loop will result in an additional number of `COLS` LD nodes generated and connected to global memory 2, and thereby an increase in the *Area Estimates* tag.

## Run the `annotated_ptr` Executable

### On Linux
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./annotated_ptr.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./annotated_ptr.fpga_sim
   ```
### On Windows
1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   annotated_ptr.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   annotated_ptr.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

## Example Output

```
Running on device: Intel(R) FPGA Emulation Device
PASSED: The results are correct
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
