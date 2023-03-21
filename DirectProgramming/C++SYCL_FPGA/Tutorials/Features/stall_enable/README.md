# Reducing latency of computations
This FPGA tutorial demonstrates how to use the `use_stall_enable_clusters` attribute to reduce the area and latency of your FPGA kernels.  This attribute may reduce the FPGA FMax for your kernels.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel® Agilex®, Arria® 10, and Stratix® 10 FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               |  What the `use_stall_enable_clusters` attribute does <br> How `use_stall_enable_clusters` attribute affects resource usage and latency <br> How to apply the `use_stall_enable_clusters` attribute to kernels in your program
| Time to complete                  | 15 minutes (emulator and report) <br> several hours (fpga bitstream generation)

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

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstrates a compiler feature.

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

## Purpose
The `use_stall_enable_clusters` attribute enables you to direct the compiler to reduce the area and latency of your kernel.  Reducing the latency will not have a large effect on loops that are pipelined, unless the number of iterations of the loop is very small.
Computations in an FPGA kernel are normally grouped into *Stall Free Clusters*. This allows simplification of the signals within the cluster, but there is a FIFO queue at the end of the cluster that is used to save intermediate results if the computation needs to stall. *Stall Enable Clusters* save area and cycles by removing the FIFO queue and passing the stall signals to each part of the computation.  These extra signals may cause the FMax to be reduced

> **Note**: If you specify `[[intel::use_stall_enable_clusters]]` on one or more kernels, this may reduce the FMax of the generated FPGA bitstream, which may reduce performance on all kernels.

> **Note**: The `use_stall_enable_clusters` attribute is not applicable for designs that target FPGA architectures that support hyper optimized handshaking, unless the `-Xshyper-optimized-handshaking=off` argument is passed to `icpx`

### Example: Using the `use_stall_enable_clusters` attribute
```
h.single_task<class KernelComputeStallFree>( [=]() [[intel::use_stall_enable_clusters]] {
  // The computations in this device kernel will use Stall Enable clusters
  Work(accessor_vec_a, accessor_vec_b, accessor_res);
});
```
The FPGA compiler will use *Stall Enable Clusters* for the kernel when possible.  Some computations may not be able to stall and need to be placed in a *Stall Free Cluster*.

## Key Concepts
* Description of the `use_stall_enable_clusters` attribute
* How `use_stall_enable_clusters` attribute affects resource usage and loop throughput
* How to apply the `use_stall_enable_clusters` attribute to kernels in your program

## Building the `stall_enable` Tutorial

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

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex® device family), run `cmake` using the command:
  ```
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

  > **Note**: for targets that support hyper optimized handshaking, it must be disabled with `-DNO_HYPER_OPTIMIZATION=true`.

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     make fpga_emu
     ```
   * Compile for simulation (fast compile time, targets simulator FPGA device):
     ```
     make fpga_sim
     ```
   * Generate the optimization reports:
     ```
     make report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     make fpga
     ```

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex® device family), run `cmake` using the command:
  ```
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

  > **Note**: for targets that support hyper optimized handshaking, it must be disabled with `-DNO_HYPER_OPTIMIZATION=true`.

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

  * Compile for emulation (fast compile time, targets emulated FPGA device):
    ```
    nmake fpga_emu
    ```
  * Generate the optimization report:
    ```
    nmake report
    ```
  * Compile for simulation (fast compile time, targets simulator FPGA device):
    ```
    nmake fpga_sim
    ```
  * Compile for FPGA hardware (longer compile time, targets FPGA device):
    ```
    nmake fpga
    ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Examining the Reports
Locate `report.html` in the `stall_enable_report.prj/reports/` and `stall_free_report.prj/reports/` directories. Open the reports in Chrome*, Firefox*, Edge*, or Internet Explorer*.

On the main report page, scroll down to the section titled `Compile Estimated Kernel Resource Utilization Summary`. Note that the estimated number of **MLAB**s for KernelComputeStallEnable is smaller than that of KernelComputeStallFree. The reduction in MLABs is due to the elimination of the FIFO queue at the end of the cluster.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./stall_enable.fpga_emu     (Linux)
     stall_enable.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA simulator device:
  * On Linux
    ```bash
    CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./stall_enable.fpga_sim
    CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./stall_free.fpga_sim
    ```
  * On Windows
    ```bash
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
    stall_enable.fpga_sim.exe
    stall_free.fpga_sim.exe
    set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
    ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`):
     ```
     ./stall_enable.fpga         (Linux)
     ./stall_free.fpga           (Linux)
     stall_enable.fpga.exe       (Windows)
     stall_free.fpga.exe         (Windows)
     ```

### Example of Output

```
PASSED: The results are correct
```

### Discussion of Results
There should be no observable difference in results in the example.

Only the area and latency differences are visible in the reports.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
