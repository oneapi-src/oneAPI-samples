# `Explicit Data Movement` Sample

An FPGA tutorial demonstrating an alternative coding style, SYCL* Unified Shared Memory (USM) device allocations, in which data movement between host and device is controlled explicitly by the code author.

| Area                 | Description
|:---                  |:---
| What you will learn  | How to manage the movement of data explicitly for FPGA devices
| Time to complete     | 15 minutes
| Category             | Code Optimization

## Purpose

The purpose of this tutorial is to demonstrate an alternative coding style that allows you to control the movement of data explicitly in your SYCL-compliant program by using SYCL USM device allocations.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 2 sample that demonstrates a design pattern.

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

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware           | Intel® Agilex® 7, Agilex® 5, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software           | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Key Implementation Details

### Implicit and Explicit Data Movement

A typical SYCL design copies input data from the Host DDR to the FPGA DDR. To do this, the data is sent to the FPGA board over PCIe. Once all the data is copied to the FPGA DDR, the FPGA kernel is run and produces output that is also stored in the FPGA DDR. Finally, the output data is transferred from the FPGA DDR back to the host DDR over PCIe. This model is shown in the figure below.

```
|------------|                               |----------|
|            |   |-------| PCIe |--------|   |          |
|  Host DDR  |<=>|  CPU  |<====>|  FPGA  |<=>| FPGA DDR |
|            |   |-------|      |--------|   |          |
|------------|                               |----------|
```

SYCL provides two methods for managing the movement of data between host and device:

- **Implicit data movement**: Copying data to and from the FPGA DDR is not controlled by the code author but rather by a parallel runtime. The parallel runtime is responsible to ensure data is correctly transferred before being used and that kernels accessing the same data are synchronized appropriately.
- **Explicit data movement**: Copying data to and from the FPGA DDR is explicitly handled by the code author in the source code. The code author is responsible to ensure data is correctly transferred before being used and that kernels accessing the same data are synchronized appropriately.

Most of the other code sample tutorials in the [FPGA Tutorials](/DirectProgramming/C++SYCL_FPGA/Tutorials) in the oneAPI-samples GitHub repository use implicit data movement, which is achieved through the `buffer` and `accessor` SYCL constructs. This tutorial demonstrates an alternative coding style using explicit data movement, which is achieved through USM allocations.

### SYCL USM Allocations

SYCL USM allocations allows users to allocate memory directly accessible by the FPGA.
There are three types of such allocations:
- device allocations (through the use of `malloc_device`). This type of allocation matches the example given above
- host allocations (through the use of `malloc_host`). The pointer given to the FPGA actually lives in the host DDR, but the BSP implements the necessary functionalities to be able to read directly in this memory.
- shared allocations (through the use of `malloc_shared`). The memory can live on the host or on the FPGA and may be moved at runtime between the host and device depending on the memory access pattern.

SYCL USM allocations enable the use of raw *C-style* pointers in SYCL. 
When copying data to or from the FPGA DDR, the data transfer must be done explicitly by the programmer, typically using the SYCL `memcpy` function. 
Additionally, synchronization between kernels accessing the same device pointers must be expressed explicitly by the programmer using either the `wait` function of a SYCL `event` or the `depends_on` signal between events.

In this tutorial, the `SubmitImplicitKernel` function demonstrates the basic SYCL buffer model, while the `SubmitExplicitKernel` function demonstrates how you may use these USM device allocations (in full stack flow) or USM host allocations (in the SYCL HLS flow) to perform the same task and manually manage the movement of data.

### Deciding Which Style to Use

The main advantage of explicit data movement is that it gives you full control over when data is transferred between different memories. In some cases, one can improve a design's overall performance by overlapping data movement and computation, which can be done with more fine-grained control using USM allocations. The main disadvantage of explicit data movement is that specifying all data movements can be tedious and error prone.

Choosing a data movement strategy largely depends on the specific application and personal preference. However, when starting a new design, it is typically easier to begin by using implicit data movement since all of the data movement is handled automatically, allowing the user to focus on expressing and validating the computation. Once the design is validated, one can:
- profile the application in the full stack flow. If there is still performance on the table, it may be worthwhile switching to explicit data movement.
- customize the IP interface in the SYCL HLS flow. One can annotate these raw pointers to direct the compiler into implementing different protocols. An overview of the different interfaces can be found in the [component_interfaces_comparison](/DirectProgramming/C%2B%2BSYCL_FPGA/Tutorials/Features/hls_flow_interfaces/component_interfaces_comparison) sample.

Alternatively, in the full stack flow, there is a hybrid approach that uses some implicit data movement and some explicit data movement. This technique, demonstrated in the **Double Buffering** (double_buffering) and  **N-Way Buffering** (n_way_buffering) tutorials, uses implicit data movement for some buffers where the control does not affect performance, and explicit data movement for buffers whose movement has a substantial effect on performance. In this hybrid approach, we do **not** use device allocations but rather specific `buffer` API calls (e.g., `update_host`) to trigger the movement of data.

## Build the `Explicit Data Movement` Sample

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
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
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
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
   >  ```
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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
      The report resides at `explicit_data_movement.prj/reports/report.html`. Note that because the optimization occurs at the *runtime* level, the FPGA compiler report will not show a difference between the optimized and unoptimized cases.

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
  >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant> -DIS_BSP=1
  >  ```
  > **Note**: You can poll your system for available BSPs using the `aoc -list-boards` command. The board list that is printed out will be of the form
  > ```
  > $> aoc -list-boards
  > Board list:
  >   <board-variant>
  >      Board Package: <path/to/board/package>/board-support-package
  >   <board-variant2>
  >      Board Package: <path/to/board/package>/board-support-package
  > ```
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
      The report resides at `explicit_data_movement.prj.a/reports/report.html`.

   4. Compile for FPGA hardware (longer compile time, targets FPGA device).
      ```
      nmake fpga
      ```

>**Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your `build` directory in a shorter path, for example `C:\samples\build`. You can then build the sample in the new location, but you must specify the full path to the build files.

## Run the `Explicit Data Movement` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./explicit_data_movement.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./explicit_data_movement.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./explicit_data_movement.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   explicit_data_movement.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   explicit_data_movement.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   explicit_data_movement.fpga.exe
   ```

## Example Output

### Output Example for FPGA Emulator

```
Running the ImplicitKernel with size=10000
Running the ExplicitKernel with size=10000
PASSED
```

### Output Example for FPGA Device

```
Running the ImplicitKernel with size=100000000
Running the ExplicitKernel with size=100000000
Average latency for the ImplicitKernel: 530.762 ms
Average latency for the ExplicitKernel: 470.453 ms
PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
