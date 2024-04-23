# Using the Intel® FPGA Dynamic Profiler for DPC++

This FPGA tutorial demonstrates how to use the Intel® FPGA Dynamic Profiler for DPC++ to dynamically collect performance data from an FPGA design and reveal areas for optimization and improvement.
> **Note**: This code sample is not yet supported in Windows*.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15
| Hardware                          | Intel® Agilex® 7, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | About the Intel® FPGA Dynamic Profiler for DPC++ <br> How to set up and use this tool <br> A case study of using this tool to identify performance bottlenecks in pipes.
| Time to complete                  | 15 minutes

> **Note**: This sample has been tuned to show the results described on Arria 10 devices. While it compiles and runs on the other supported devices, the hardware profiling results may differ slighly from what is described below.

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.
>
> :warning: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Prerequisites

This sample is part of the FPGA code samples.
It is categorized as a Tier 3 sample that demonstrates the usage of a tool.

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

This FPGA tutorial demonstrates how to use the Intel® FPGA Dynamic Profiler for DPC++ to collect performance data from a design running on FPGA hardware and how to use the Intel® VTune™ Profiler to analyze the collected data. This tutorial focuses on optimizing the usage of loops and pipes within an FPGA design using stall and occupancy kernel performance data.

###  Profiling Tools

Intel® oneAPI provides two runtime profiling tools to help you analyze your SYCL design for FPGA:

1. The **Intel® FPGA Dynamic Profiler for DPC++** is a profiling tool used to collect fine-grained device side data during SYCL* kernel execution. When used within the Intel® VTune™ Profiler, some host side performance data is also collected. However, note that the VTune Profiler is not designed to collect detailed system level host-side data.

2. The **Intercept Layer for OpenCL™ Applications™** is a profiling tool used to obtain detailed system-level information.
This tutorial introduces the Intel® FPGA Dynamic Profiler for DPC++. (To learn more about the Intercept Layer, refer to the [Using the Intercept Layer for OpenCL™ Applications to Identify Optimization Opportunities](/DirectProgramming/C++SYCL_FPGA/Tutorials/Tools/system_profiling) FPGA tutorial.)

#### The Intel® FPGA Dynamic Profiler for DPC++

Work-item execution stalls can occur at various stages of a kernel pipeline. Applications that use many pipes or memory accesses might frequently stall to enable the completion of memory transfers. The Dynamic Profiler collects various performance metrics such as stall, occupancy, idle, and bandwidth data at these points in the pipeline to make it easier to identify the memory or pipe operations, creating stalls.

Consider the following program:

```c++
// Vector Add Kernel
h.single_task<VectorAdd>([=]() {
  for (int i = 0; i < kSize; ++i) {
    r[i] = a[i] + b[i];
  }
});
```
During the above program's compilation, the Dynamic Profiler adds performance counters to the three memory operations present. Then during the execution of the program, these counters measure:
- how often the operation prevents the execution pipeline from continuing (stall)
- the number of valid reads/writes occurring (occupancy)
- how often the memory access point is unused (idle)
- how much data gets written/read from memory at this access (bandwidth).

Once the run completes, the data is stored in a `profile.json` file and can be displayed using the Intel® VTune™ Profiler.

### Using the Intel® FPGA Dynamic Profiler for DPC++

#### Enabling Dynamic Profiling During Compilation

The Intel® FPGA Dynamic Profiler for DPC++ comes as part of the Intel® oneAPI Base Toolkit.

To instrument the kernel pipeline with performance counters, add the `-Xsprofile` flag to your FPGA hardware compile command.

For this tutorial, the `-Xsprofile` flag has already been added to the cmake command for your convenience.

#### Obtaining Dynamic Profiling Data During Runtime

There are two ways of obtaining data from a program containing performance counters:

1. Run the design in the Intel® VTune™ Profiler via the CPU/FPGA Interaction viewpoint.

    Instructions on installing, configure and opening the Intel® VTune™ Profiler can be found in the [Intel® VTune™ Profiler User Guide](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/installation.html). Further instructions on setting up the Dynamic Profiler via the CPU/FPGA Interaction View can be found in the [CPU/FPGA Interaction Analysis](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/accelerators-group/cpu-fpga-interaction-analysis-preview.html) section of the Intel® VTune™ Profiler User Guide. To extract device performance counter data, please ensure the source for the FPGA profiling data is set to "AOCL Profiler".

2. Run the design from the command line using the Profiler Runtime Wrapper.
  The Profiler Runtime Wrapper comes as part of the Intel® oneAPI DPC++/C++ Compiler and can be run as follows:
   ```
   aocl profile <executable>
   ```
   More details and options can be found in the "[Invoke the Profiler Runtime Wrapper to Obtain Profiling Data](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/analyze-your-design/analyze-the-fpga-image/intel-fpga-dynamic-profiler-for-dpc/obtain-profiling-data-during-runtime/invk-prof-rt-wrpr.html)" section of the oneAPI FPGA Optimization Guide.

    Upon execution completion, a `profile.json` file will have been generated. This file contains all of the data collected by the Dynamic Profiler. It can either be viewed manually or imported into the Intel® VTune™ Profiler, as explained in [Import Results and Traces into VTune Profiler GUI](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/manage-result-files/importing-results-to-gui.html).

To run this tutorial example, refer to the "[Running the Sample](#running-the-sample)" section.

#### Viewing the Performance Data

After running the executable and opening the `profile.json` file in the Intel® VTune™ Profiler (done automatically when run in the VTune Profiler), the VTune Profiler will display the performance results collected from the SYCL kernel pipelines.

The CPU/FPGA Interaction viewpoint is comprised of four windows:
- The **Summary window** displays statistics on the overall application execution. It can be used to identify CPU time and processor utilization, and the execution time for SYCL kernels.
- The **Bottom-up window** displays details about the kernels in a bottom-up tree. This tree contains computing task information such as total time and instance count and device metrics such as overall stall, occupancy, and bandwidth. This data is also displayed in a timeline.
- The **Platform window** displays over-time performance data for the kernels, memory transfers, CPU context switches, FPU utilization, and CPU threads with FPGA kernels.
- The **Source window** displays device-side data at the corresponding source line. Unlike the other three windows, which are accessible via tabs, the Source window is opened on a per-kernel basis by double-clicking the kernel name in the Bottom-up view.

### Understanding the Performance Metrics

For this tutorial, the focus will be on a few specific types of device metrics obtained by the Dynamic Profiler. For explanation and more information on other device metrics obtained by the Dynamic Profiler and the Intel® VTune™ Profiler please review the [FPGA Optimization Guide for Intel® oneAPI Toolkits Developer Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) and the [Intel® VTune™ Profiler User Guide](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html).

For additional design scenario examples demonstrating how to use other performance data for optimization, refer to [Profiler Analyses of Example SYCL* Design Scenarios](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/analyze-your-design/analyze-the-fpga-image/intel-fpga-dynamic-profiler-for-dpc/profiler-analyses-of-example-dpc-design-scenarios.html).

#### FPGA Metrics for this Tutorial

The following device metrics are important in this tutorial:

| Device Metric                     | Description
|:---                               |:---
| Pipe Stall %                      | The percentage of time the pipe is causing the pipeline to stall. It measures the ability of the pipe to fulfill an access request.
| Pipe Occupancy %                  | The percentage of time when a valid work-item executes a pipe access.

When analyzing performance data to optimize a design, the goal is to get as close to an ideal kernel pipeline as possible. Pipes in an ideal pipeline should have no stalls, occupancy of 100%, and an average pipe depth close to the maximum possible depth. If memory or pipe access is far from the ideal, it may be a candidate for an investigation into what went wrong. For example, if the stall percentage of a pipe write is high, it may indicate that the kernel controlling the pipe's read side is slower than the write side kernel, causing the pipe to fill up, leading to a performance bottleneck.

#### Analyzing Stall and Occupancy Metrics

In this tutorial, there are two design scenarios defined in dynamic_profiler.cpp. One showing a naive pre-optimized design, and a second showing the same design optimized based on data collected through the Intel® FPGA Dynamic Profiler for DPC++ on an Arria 10 device.

##### Pre-optimization Version #####

The first scenario contains two kernels:
- a producer SYCL kernel (ProducerBefore) that reads data from a buffer and writes it to a pipe (ProducerToConsumerBeforePipe), and
- a consumer SYCL kernel (ConsumerBefore) that reads data from the pipe (ProducerToConsumerBeforePipe), performs two computation operations on the pipe data and then outputs the result to a buffer.

After compiling and running dynamic_profiler.cpp with the Dynamic Profiler (see [Running the Sample](#running-the-sample) for more information), open the Bottom-Up window of the CPU/FPGA Interaction viewpoint in the Intel® VTune™ Profiler to view the overall kernel performance data. You should see that ProducerBefore's pipe write (on line 68) has a stall percentage of nearly 100%. This shows that the ProducerBefore kernel is writing so much data to the pipe that it is filling up and stalling. From this, it can be concluded that the consumer kernel is reading data more slowly than the producer is writing it, causing the bottleneck. This is likely because the consumer kernel is performing several compute heavy operations on the data from the pipe, causing a large time delay before the next datapoint can be read. Recalling the ideal kernel described earlier, this would indicate that this design is not optimized and can possibly be improved.

##### Post-optimization Version #####

The second scenario is an example of what the design might look like after being optimized to reduce the high stall values. Like before, the design contains two kernels:
- a producer SYCL kernel (ProducerAfter) that reads data from a buffer, performs the first computation on the data and writes this value to a pipe (ProducerToConsumerAfterPipe), and
- a consumer SYCL kernel (ConsumerAfter) that reads from the pipe (ProducerToConsumerAfterPipe), does the second set of computations and fills up the output buffer.

When looking at the performance data for the two "after optimization" kernels in the Bottom-Up view, you should see that ProducerAfter's pipe write (on line 126) and the ConsumerAfter's pipe read (line 139) both have stall percentages near 0%. This indicates the pipe is being used more effectively - now the read and write side of the pipe are being used at similar rates, so the pipe operations are not creating stalls in the pipeline. This also speeds up the overall design execution - the two "after" kernels take less time to execute than the two before kernels.

![](profiler_pipe_tutorial_bottom_up.png)

![](profiler_pipe_tutorial_source_window.png)

## Key Concepts

- A summary of profiling tools available for performance optimization
- How to use the FPGA Dynamic Profiler
- How to set up and use this tool within the Intel® VTune™ Profiler
- How to use performance data to identify performance bottlenecks in a design's kernel pipeline

## Building the Tutorial

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

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
  ```
  mkdir build
  cd build
  ```
  To compile for the default target (the Agilex® 7 device family), run `cmake` using the command:
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

2. Compile the design through the generated `Makefile`. The following build targets are provided:
   * Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):

    ```bash
    make fpga_sim
    ```
   *  Compile for FPGA hardware (longer compile time, targets an FPGA device) using:

    ```bash
    make fpga
    ```

## Running the Sample

To collect dynamic profiling data, choose one of the following methods:

**Within the Intel® VTune™ Profiler (Recommended)**

1. Open the Intel® VTune™ Profiler on a machine with installed and configured FPGA hardware.

2. Create a new VTune Profiler project.

3. In the Configure Analysis tab, configure a Dynamic Profiler run by:
   - setting "How" pane to "CPU/FPGA Interaction"
   - in the "How" pane, changing the "FPGA profiling data source" to "AOCL Profiler"
   - in the "What" pane, set the application to your compiled FPGA hardware executable (dynamic_profiler.fpga)

4. Run the Dynamic Profiler by clicking the run button in the lower right corner (below the "How" pane). The VTune Profiler will open up the profiling results once the execution finishes.

![](profiler_pipe_tutorial_configure_vtune.png)

**At the Command Line**
1. Run the design using the makefile targets generated in "[On a Linux* System](#on-a-linux-system)":
    * Run the design using the simulator:
      ```
      export CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
      make run_sim
      unset CL_CONTEXT_MPSIM_DEVICE_INTELFPGA
      ```
    * Run the design on hardware (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`):
      ```
      make run
      ```
    These targets run the executable with the Profiler Runtime Wrapper, creating a `profile.json` data file in the current directory.

2. Open the Intel® VTune™ Profiler.

3. Follow the instructions in [Import Results and Traces into VTune Profiler GUI](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/manage-result-files/importing-results-to-gui.html) to import the `profile.json` file. The VTune Profiler will open up the profiling results once the import completes.

### Example of Output

The expected command line output after running the design is
```
Setting profiler start cycle from environment variable: 1
*** Beginning execution before optimization.
Verification PASSED for run before optimization
*** Beginning execution after optimization.
Verification PASSED for run after optimization
Verification PASSED
Starting post-processing of the profiler data...
Post-processing complete.
```

> **Note**: You can ignore warnings about pipe data resulting in a division by zero. These are the result of pipes not yet receiving data resulting in metrics being zero.

An example output, `dynamic_profiler_tutorial.json`, can be found zipped in the `data` folder. Without needing to run the design, you can import this data file in the VTune Profiler, as described in the [Import Results and Traces into VTune Profiler GUI](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/manage-result-files/importing-results-to-gui.html) documentation, to observe the kind of data the Dynamic Profiler can collect.

## Examining the Reports
In the tab containing the data from your run, ensure the view is set to "CPU/FPGA Interaction". There should be three windows for viewing:

1. Summary
2. Bottom-up
3. Platform

This tutorial focuses on the Bottom-Up window. To navigate there, click on the Bottom-Up window and set the grouping to "Computing Task / Channel / Compute Unit". Initially, all profiling data in the Bottom-Up and Source windows will be summed over all SYCL kernel runs. To view data from a particular point in a kernel's run, click and drag to select a portion of data in the execution timeline in the lower half of the Bottom-Up window and choose "Filter In by Selection". Then all data in the table or Source window will only be from the current selected timeslice.

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
