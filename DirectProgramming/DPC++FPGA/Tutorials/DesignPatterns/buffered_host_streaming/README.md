# Buffered Host-Device Streaming
This tutorial demonstrates how to create a high-performance full system CPU-FPGA design using SYCL USM.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX)
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | How to optimally stream data between the host and device to maximize throughput
| Time to complete                  | 45 minutes

_Notice: SYCL USM host allocations (and therefore this tutorial) are only supported for the Intel&reg; FPGA PAC D5005 (with Intel Stratix 10 SX)_
_Notice: This tutorial demonstrates an implementation of host streaming that will be supplanted by better techniques in a future release. See the [Drawbacks and Future Work](#drawbacks-and-future-work)_

## Purpose
This tutorial demonstrates how to optimize a full system design that streams data from the host, to the device, and back to the host. This tutorial illustrates techniques to create heterogeneous designs that can achieve high throughput. The techniques described in this tutorial are not specific to a CPU-FPGA system (like the one used in this tutorial); they apply to GPUs, multi-core CPUs, and other processing units.

Before starting this tutorial, we recommend first reviewing the **Simple Host Streaming** (simple_host_streaming), **Double Buffering** (double_buffering), and **N-Way Buffering** (n_way_buffering) FPGA tutorials. The concepts explained in these tutorials will be used in this tutorial to create a highly optimized heterogeneous design. This tutorial also assumes that the reader has a basic understanding of multi-threaded C++ programming. More information on C++ multi-threading programming can be found [here](http://www.cplusplus.com/reference/multithreading/).

### Overview
In this tutorial, we will create a design where a *Producer* (running on the CPU) produces data into USM host allocations, a *Kernel* (running on the FPGA) processes this data and produces output into host allocations, and a *Consumer* (running on the CPU) consumes the data. Data is shared between the host and FPGA device via host pointers (pointers to USM host allocations).

![](block_diagram.png)

In heterogeneous systems, it is important to understand how the different compute architectures are connected (i.e. how data is transferred from one to the other) to better reason about and address performance bottlenecks in the system. The figure below is a slightly more detailed illustration of the processing pipeline in this tutorial, which shows the data flow between the *Producer*, *Kernel*, and *Consumer*. It illustrates that the *Producer* and *Consumer* operations both execute on the CPU and communicate with the FPGA kernel over a PCIe link (PCIe Gen3 x16).

![](block_diagram_detailed.png)

### Roofline Analysis
When designing for throughput, the system developer should first estimate the maximum throughput the system is capable of. This back-of-the-envelop calculation is called *roofline analysis*. What is the maximum achievable throughput of the Producer-Kernel-Consumer design of the previous section?

To start, let's assume the *Producer* and *Consumer* have infinite bandwidth (i.e. they can produce/consume data at an infinite rate, respectively). Then, the bottleneck in the system will be the FPGA kernel. Let's say the kernel has an f<sub>MAX</sub> of 400MHz and processes 64 bytes per cycle. The kernel's maximum steady-state throughput is 400 MHz * 64 bytes/cycle = 25.6 GB/s ~= 26 GB/s.

In reality, the rate of data transfer to and from the kernel depends on the bandwidth of the PCIe link. Looking at the previous figure again, we see that this depends on the bandwidth of the PCIe link. We have experimentally measured the PCIe Gen3 x16 link to have a total bandwidth of ~22 GB/s (11 GB/s bi-directional). This means that, while our kernel can process data at 26 GB/s, we can only get data to it and from it at a rate of 11 GB/s. The system bottleneck is the PCIe link between the CPU and FPGA. It is common for the bottleneck in a heterogenous system to be the inter-device link, rather than any single device.

In the analysis we just performed, we assumed that the *Producer* and *Consumer* had infinite bandwidth. We can perform a roofline analysis on the entire system now by assuming a finite bandwidth for the *Producer* and *Consumer*. For the entire *Producer*-*Kernel*-*Consumer* design, the *maximum* possible throughput is the *minimum* of the *Producer*, *Kernel*, and *Consumer* throughput, and the bandwidth of the link (in this case, PCIe) connecting the CPU and FPGA; "*A chain is only as strong as the weakest link*".

A roofline analysis is performed in the `DoRooflineAnalysis` function in *buffered_host_streaming.cpp*.

### Optimizing the Design
This section discusses how to optimize the design for throughput.

The naive approach to this design is for the *Producer*, *Kernel*, and *Consumer* to run sequentially. The timing diagram for this approach is shown in the figure below. This approach under-performs because operations, which could run in parallel, are run sequentially. For example, in the figure below, the *Producer* could be producing the second set of data, while the *Consumer* is consuming the first set of data.

![](timing_diagram_basic_0.png)

To improve the performance, we can create a separate CPU thread (called the *Producer thread*) which produces all of the data for the *Kernel* to process, and be later consumed. As shown in the figure below, this allows the *Producer* and *Consumer* to run in parallel (i.e. the *Producer* thread produces the next set of data, while the *Consumer* consumes the current data). This approach improves the design's throughput, but does require more complicated thread synchronization logic to synchronize between the *Producer* thread and the main thread that is launching kernels and consuming its output.

Notice that in the figure below the *Producer* and *Consumer* are running simultaneously (in separate threads). It is important to account for the fact that both of these processes are running on the CPU in parallel, which affects their individual throughput capabilities. For example, running these processes in parallel could result in saturating the CPU's memory bandwidth, and therefore lower the overall throughput for the *Producer* and *Consumer* processes; even if the threads run on different CPU cores! We account for this in `DoRooflineAnalysis` function by running the *Producer* and *Consumer* in isolation, and in parallel.

![](timing_diagram_basic_1.png)

By putting the *Producer* into its own thread, we are able to improve performance by producing and consuming data simultaneously. However, it would be ideal if the *Producer* could produce the next set of data while the kernel is processing the current data. This would allow the next kernel to launch as soon as the current kernel finishes, instead of waiting for the *Producer* to finish, like in the figure above. Unfortunately, we cannot produce data into the buffer that the kernel is currently processing, since it is still in use by the kernel.

To address this, we use multiple buffers (i.e. n-way buffering) of the input and output buffers, which results in a timing diagram like the figure below. The subscript number on the *Producer* and *Consumer* (e.g. *Producer*<sub>0</sub> and *Consumer*<sub>0</sub>) represents the buffer index they are processing. The figure below illustrates the case where we use 2 sets of buffers (i.e. double-buffering). Notice that in the figure below, the *Producer* produces into buffer 1 (*Producer*<sub>1</sub>) while the kernel is processing the data from buffer 0. Thus, by the time the kernel finishes processing buffer 0, it can start processing buffer 1 right away, so long as the *Producer* has finished producing the next buffer of data.

![](timing_diagram_basic_double_buffered.png)

In the steady-state for the figure above, **the throughput of the design is bottlenecked by the throughput of the slowest stage** (the *Producer*, *Kernel*, or *Consumer*), which matches our roofline analysis from earlier! For example, the first figure below shows an example timeline where the *Kernel* is the throughput bottleneck. The second image below shows an example timeline where the *Producer* (or *Consumer*) is the bottleneck.

![](timing_diagram_basic_double_buffered_kernel_bottleneck.png)

![](timing_diagram_basic_double_buffered_cpu_bottleneck.png)

### Streaming API
In this tutorial, the technique described in the previous section is implemented in two ways. The design is implemented directly using SYCL USM host allocations, C++ multi-threading, and intelligent management of the SYCL kernel queue. The code achieves high performance, but it may be difficult to understand and extend to different designs. To address this, we have created a convenient and performant API wrapper (`HostStreamer.hpp`). The same design is implemented in `streaming_with_api.hpp` with similar performance and significantly less code that is much easier to understand.

**An important note on performance:** While the code that uses the `HostStreamer` API achieves similar performance to a direct implementation, it uses extra FPGA resources. The direct implementation has a single kernel (*Kernel*) that does all of the processing. Using the API creates a *Producer* and *Consumer* kernel that access host allocations and produce/consume data to/from the processing kernel (`APIKernel` in `streaming_with_api.hpp`). These extra kernels (that are transparent to the user) are the mechanism by which the API abstracts the production/consumption of data, but come at the cost of extra FPGA resources. However, when compiled for the Intel Stratix&reg; 10 SX, these extra kernels result in less than a 1% increase in FPGA resource utilization. Therefore, given the programming convenience they provide, the tradeoff is often worth it.

### Drawbacks and Future Work
Fundamentally, the ability to stream data between the host and device is built around USM host allocations. The underlying problem is how to efficiently synchronize between the host and device to signal that _some_ data is ready to be processed, or has been processed. In other words, how does the host signal to the device that some data is ready to be processed? Conversely, how does the device signal to the host that some data is done being processed?

In this tutorial, we perform this synchronization by using the start of a kernel to signal to the device that the host has produced data to be processed. We use the end of a kernel to signal to the host that data has been processed by the device. However, this method has notable drawbacks.

First, the latency to start and end kernels is high (as of now, roughly 50us). To maintain performance, we must size the USM host allocations sufficiently large to hide the inter-kernel latency. This sacrifices design latency and memory usage for the multiple input and output host allocations.

Second, the programming model to achieve this performance is non-trivial. It involves double-buffering, thread synchronization to avoid race conditions, and intelligently managing the SYCL device queue to efficiently launch and wait on the finish kernels without increasing inter-kernel latency and decreasing throughput. In this tutorial, we partially addressed this drawback with an API (`HostStreamer.hpp`) that hides this logic from the user. This API uses extra FPGA resources and extra CPU threads that are necessary to cleanly abstract the complicated double-buffering and SYCL queue management logic, but are not necessary to achieve the desired performance (i.e. you could get the same performance without the API).

We are currently working on a new API and tutorial to address these drawbacks. This API will decrease the latency to synchronize between the host and device, use less area and CPU threads than the `HostStreamer.hpp` API, and will dramatically improve the usability of the programming model to achieve the desired performance.

## Key Concepts
* Host-device optimizations for high performance heterogeneous designs
* Double-buffering
* Run time SYCL kernel management
* C++17 Multi-threaded programming
 
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `buffered_host_streaming` Tutorial

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile, fpga_runtime:arria10, or fpga_runtime:stratix10) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.
 
### On a Linux* System
 
1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:
    ```
    cmake ..
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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/buffered_host_streaming.fpga.tar.gz" download>here</a>.
 
### On a Windows* System
 
1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:  
    ```
    cmake -G "NMake Makefiles" ..
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
   * An FPGA hardware target is not provided on Windows*. 
 
*Note:* The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.
 
 ### In Third-Party Integrated Development Environments (IDEs)
 
You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel&reg; oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `buffered_host_streaming_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

## Running the Sample
 
 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./buffered_host_streaming.fpga_emu     (Linux)
     buffered_host_streaming.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./buffered_host_streaming.fpga         (Linux)
     ```
 
### Example of Output
The following results were obtained on a system with the following specifications:</br>
*CPU*: Intel&reg; Xeon&reg; CPU E5-1650 v3 @ 3.50GHz (6 cores, 12 threads) </br>
*CPU Memory*: 65GB </br>
*Accelerator*: Intel&reg; PAC D5005 (with Intel Stratix&reg; 10 SX FPGA) </br>
*PCIe*: Gen3 x16 </br>

You should see the following output in the console:

1. When running on the FPGA emulator
    ```
    Repetitions:      20
    Buffers:          2
    Buffer Count:     4096
    Iterations:       1
    Total Threads:    12

    Running the roofline analysis
    Producer (6 threads)
      Time:       0.5451 ms
      Throughput: 480.9159 MB/s
    Consumer (6 threads)
      Time:       0.3505 ms
      Throughput: 747.8360 MB/s
    Producer & Consumer (6 threads, each)
      Time:       1.8754 ms
      Throughput: 139.7806 MB/s
    Kernel
      Time:       0.5632 ms
      Throughput: 465.4248 MB/s

    Maximum Design Throughput: 139.7806 MB/s
    The FPGA kernel does not limit the performance of the design
    Done the roofline analysis

    Running the full design without API
    Average latency without API: 1.5041 ms
    Average processing time without API: 29.5807 ms
    Average throughput without API: 177.2398 MB/s

    Running the full design with API
    Average latency with API: 5.0432 ms
    Average processing time with API: 76.0256 ms
    Average throughput with API: 68.9620 MB/s

    PASSED
    ```
    NOTE: The FPGA emulator does not accurately represent the performance (throughput or latency) of the kernels.

2. When running on the FPGA device
    ```
    Repetitions:      200
    Buffers:          2
    Buffer Count:     524288
    Iterations:       4
    Total Threads:    12

    Running the roofline analysis
    Producer (6 threads)
            Time:       2.1712 ms
            Throughput: 15453.9737 MB/s
    Consumer (6 threads)
            Time:       2.1064 ms
            Throughput: 15929.9736 MB/s
    Producer & Consumer (6 threads, each)
            Time:       4.2813 ms
            Throughput: 7802.4714 MB/s
    Kernel
            Time:       3.2528 ms
            Throughput: 10315.6158 MB/s

    Maximum Design Throughput: 7802.4714 MB/s
    The FPGA kernel does not limit the performance of the design
    Done the roofline analysis

    Running the full design without API
    Average latency without API: 4.8865 ms
    Average processing time without API: 979.9290 ms
    Average throughput without API: 6829.5611 MB/s

    Running the full design with API
    Average latency with API: 3.3371 ms
    Average processing time with API: 1053.3818 ms
    Average throughput with API: 6370.8013 MB/s

    PASSED
    ```
    NOTE: In the performance results above, the FPGA kernel is **not** the bottleneck of the full system; the *Producer*/*Consumer* running in parallel are. The full design achieves ~87% of the maximum possible throughput (as measured by the roofline analysis).
