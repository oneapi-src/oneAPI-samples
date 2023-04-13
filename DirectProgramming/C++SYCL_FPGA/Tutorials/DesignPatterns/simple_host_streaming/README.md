# `Host-Device Streaming using USM` Sample

This sample demonstrates how to use SYCL* Universal Shared Memory (USM) to stream data between the host and FPGA device and achieve low latency while maintaining throughput.

| Area                 | Description
|:--                   |:--
| What you will learn  | How to achieve low-latency host-device streaming while maintaining throughput
| Time to complete     | 45 minutes
| Category             | Code Optimization

## Purpose

The purpose of this tutorial is to show you how to take advantage of SYCL USM host allocations and zero-copy host memory to implement a streaming host-device design with low latency and high throughput. Before starting this tutorial, we recommend first reviewing the **Pipes** (pipes) and **Zero-Copy Data Transfer** (zero_copy_data_transfer) FPGA tutorials, which will teach you more about SYCL pipes and SYCL USM and zero-copy data transfers, respectively.

This tutorial includes three designs:

1. An offload design that maximizes throughput with no optimization for latency (`DoWorkOffload` in `simple_host_streaming.cpp`).
2. A single-kernel design that uses the methods described below to achieve a much lower latency while maintaining throughput (`DoWorkSingleKernel` in `simple_host_streaming.cpp` and `single_kernel.hpp`).
3. A multi-kernel design that uses the methods described below to achieve a much lower latency while maintaining throughput (`DoWorkMultiKernel` in `simple_host_streaming.cpp` and `multi_kernel.hpp`).

>**Note**: This tutorial demonstrates an implementation of host streaming that will be supplanted by better techniques in a future release. See the [Drawbacks and Future Work](#drawbacks-and-future-work) section below.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler


> **Note**: Even though the Intel® oneAPI DPC++/C++ Compiler is enough to compile for emulation, generating reports, generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, you must have Intel® Quartus® Prime Pro Edition and one of the following simulators installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

>**Notice**: SYCL USM host allocations, which are used in this sample, are only supported on FPGA boards that have a USM capable BSP (For example, the Intel® FPGA PAC D5005 with Intel Stratix® 10 SX with USM support: **intel_s10sx_pac:pac_s10_usm**) or when targeting an FPGA family/part number.

This sample is part of the FPGA code samples. It is categorized as a Tier 3 sample that demonstrates a design pattern.

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.

## Key Implementation Details

### Offload Processing

Typical SYCL designs perform _offload processing_. All of the input data is prepared by the CPU, and then transferred to the device (in our case, an FPGA). Kernels are started on the device to process the data. When the kernels finish, the CPU copies the output data from the FPGA back to its memory. Memory synchronization is achieved on the host after the device kernel signals its completion.

Offload processing achieves excellent throughput when the memory transfers and kernel computation are performed on large data sets, as the CPU's kernel management overhead is minimized. Data transfer overhead can be concealed using *double buffering* or *n-way buffering* to maximize kernel throughput. However, a significant shortcoming of this design pattern is latency. The coarse grain synchronization of waiting for the entire set of data to processed results in a latency that is equal to the processing time of the entire data.

This tutorial will demonstrate a simple host-device streaming design that reduces latency and maintains throughput.

### Host-Device Streaming Processing

The method for achieving lower latency between the host and device is to break data set into smaller chunks and, instead of enqueuing a single long-running kernel, launch a set of shorter-running kernels. Together, these shorter-running kernels process the data in smaller batches. As memory synchronization occurs upon kernel completion, this strategy makes the output data available to the CPU in a more granular way. This is illustrated in the figure below. The red lines show the time when the first set of data is available in the host.

![](assets/single-kernel.png)

In the streaming version, the first piece of data is available in the host earlier than in the offload version. How much earlier? Say we have `total_size` elements of data to process and we break the computation into `chunks` chunks of size `chunk_size=total_size/chunks` (as is the case in the figure above). Then, in a perfect world, the streaming design will achieve a latency that is `chunks` times better than the offload version.

#### Setting the `chunk_size`

Why not set `chunk_size` to 1 (i.e. `chunks=total_size`) to minimize the latency? In the figure above, you may notice small gaps between the kernels in the streaming design (e.g. between K<sub>0</sub> and K<sub>1</sub>). This is caused by the overhead of launching kernels and detecting kernel completion on the host. These gaps increase the total processing time and therefore decrease the throughput of the design (i.e. compared to the offload design, it takes more time to process the same amount of data). If these gaps are negligible, then the throughput is negligibly affected.

In the streaming design, the choice of the `chunk_size` is thus a tradeoff between latency (a smaller chunk size results in a smaller latency) and throughput (a smaller chunk size increases the relevance of the inter-kernel latency).

#### Lower Bounds on Latency

Lowering the `chunk_size` can reduce latency, sometimes at the expense of throughput. However, even if you aren't concerned with throughput, there still exists a lower-bound on the latency of a kernel. In the figure below, t<sub>launch</sub> is the time for a kernel launch signal to go from the host to the device, t<sub>kernel</sub> is the time for the kernel to execute on the device, and t<sub>finish</sub> is the time for the finished signal to go from the device to the host. Even if we set `chunk_size` to 0 (i.e. launch a kernel that does *nothing*) and therefore t<sub>kernel</sub> ~= 0, the latency is still t<sub>launch</sub> + t<sub>finish</sub>. In other words, the lower bound on kernel latency is the time needed for the "start "signal to get from the host to the device and for the **finished** signal to get from the device to the host.

![](assets/kernel-rtt.png)

In the previous section, we discussed how gaps between kernel invocations can degrade throughput. In the figure above, there appears to be a minimum t<sub>launch</sub> + t<sub>finish</sub> gap between kernel invocations. This is illustrated as the *Naive Relaunch* timeline in the figure below. Fortunately, this overhead is circumvented by an automatic runtime kernel launch scheme that buffers kernel arguments on the device before the previous kernel finishes. This enables kernels queue **on the device** and to begin execution without waiting for the previous kernel's "finished" to propagate back to the host. We call this *Fast Kernel Relaunch* and it is also illustrated in the figure below. While the details of fast kernel relaunch are beyond the scope of this tutorial, it is enough to understand that it reduces the gap between kernel invocations and allows you to achieve lower latency without reducing throughput.

![](assets/kernel-relaunch.png)

#### Multiple Kernel Pipeline

More complicated FPGA designs often instantiate multiple kernels connected by SYCL pipes (for examples, see the FPGA Reference Designs). Suppose you have a kernel system of `N` kernels connected by pipes, as in the figure below.

![](assets/multi-kernel-pipeline.png)

With the goal of achieving lower latency, you use the technique described in the previous section to launch multiple invocations of your `N` kernels to process chunks of data. This would give you a timeline like the figure below.

![](assets/multi-kernel.png)

Notice the gaps between the start times of the `N` kernels for a single `chunk`. This is the t<sub>launch</sub> time discussed in the previous section. However, the multi-kernel design introduces a potential new lower bound on the latency for a single chunk because processing a single chunk of data requires launching `N` kernels, which takes `N` x t<sub>launch</sub>. If `N` (the number of kernels in your system) is sufficiently large, this will limit your achievable latency.

For designs with `N > 2`, a different approach is recommended. The idea is to enqueue your system of `N` kernels **once**, and to introduce Producer (`P`) and Consumer (`C`) kernels to handle the production and consumption of data from and to the host, respectively. This method is illustrated in the figure below. The Producer streams data from the host and presents it to the kernel system through a SYCL pipe. The output of the kernel system is consumed by the Consumer and written back into host memory.

![](assets/multi-kernel-producer-consumer-pipeline.png)

To achieve low latency, we still process the data in chunks, but instead of having to enqueue `N` kernels for each chunk, we only have to enqueue a single Producer and Consumer kernel per chunk. This enables us to reduce the lower bound on the latency to MAX(2 x t<sub>launch</sub>, t<sub>launch</sub> + t<sub>finish</sub>). Notice that this lower bound does not depend on the number of kernels in our system (`N`).

**This method should only be used when `N > 2`**. FPGA area is sacrificed to implement the Producer and Consumer and their pipes in order to achieve lower overall processing latency.

![](assets/multi-kernel-producer-consumer.png)

### Drawbacks and Future Work

Fundamentally, the ability to stream data between the host and device is built around SYCL USM host allocations. The underlying problem is how to efficiently synchronize between the host and device to signal that _some_ data is ready to be processed, or has been processed. In other words, how does the host signal to the device that some data is ready to be processed? Conversely, how does the device signal to the host that some data is done being processed?

One method to achieve this signaling is to use the start of a kernel to signal to the device that data is ready to be processed, and the end of a kernel to signal to the host that data has been processed. This is the approach taken in this tutorial. However, this method has two notable drawbacks. First, the latency to start and end kernels is high (as of now, roughly 50us). To maintain high throughput, we must size the `chunk_size` sufficiently large to hide the inter-kernel latency, resulting in a latency increase. Second, the programming model to achieve this performance is non-trivial as you must intelligently manage the SYCL device queue.

We are currently working on an API and tutorial to address both of these drawbacks. This API will decrease the latency to synchronize between the host and device and therefore enable lower latency with maintained throughput. It will also dramatically improve the usability of the programming model to achieve this performance.

## Build the `Host-Device Streaming using USM` Sample

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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
2. Build the program for Intel® Agilex® 7 device family, which is the default.
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
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

    1. Compile for emulation (fast compile time, targets emulated FPGA device):
       ```
       make fpga_emu
       ```
    2. Generate the optimization report:
       ```
       make report
       ```
      The report resides at `simple_host_streaming_report.prj/reports/report.html`.

    3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
       ```
       make fpga_sim
       ```
    4. Compile for FPGA hardware (longer compile time, targets FPGA device):
       ```
       make fpga
       ```

### On Windows*

1. Change to the sample directory.
2. Build the program for the Intel® Agilex® 7 device family, which is the default.
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
   >
   > You will only be able to run an executable on the FPGA if you specified a BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report:
      ```
      nmake report
      ```
      The report resides at `simple_host_streaming_report.prj.a/reports/report.html`.

   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Host-Device Streaming using USM` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./simple_host_streaming.fpga_emu
   ```
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./simple_host_streaming.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./simple_host_streaming.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   simple_host_streaming.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   simple_host_streaming.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   simple_host_streaming.fpga.exe
   ```

## Example Output

### Example Output for FPGA Emulator

```
# Chunks:             16
Chunk count:          256
Total count:          4096
Iterations:           1

Running the basic offload kernel
Offload average latency:          0.1892 ms
Offload average throughput:       1385.3707 MB/s

Running the latency optimized single-kernel design
Single-kernel average latency:          0.0447 ms
Single-kernel average throughput:       188.3596 MB/s

Running the latency optimized multi-kernel design
Multi-kernel average latency:          0.2674 ms
Multi-kernel average throughput:       39.0021 MB/s

PASSED
```

>**Note**: The FPGA emulator does not accurately represent the performance (throughput or latency) of the kernels.

### Example Output for Intel® FPGA PAC D5005 with Intel Stratix® 10 SX with USM Support

```
# Chunks:             512
Chunk count:          32768
Total count:          16777216
Iterations:           4

Running the basic offload kernel
Offload average latency:          99.6709 ms
Offload average throughput:       107772.8673 MB/s

Running the latency optimized single-kernel design
Single-kernel average latency:          0.2109 ms
Single-kernel average throughput:       10689.9578 MB/s

Running the latency optimized multi-kernel design
Multi-kernel average latency:          0.2431 ms
Multi-kernel average throughput:       10674.7123 MB/s

PASSED
```

>**Note**: The experimentally measured bandwidth of the PCIe is ~11 GB/s (bidirectional, ~22 MB/s total). The FPGA device performance numbers above show that the offload, single-kernel, and multi-kernel designs are all able to saturate the PCIe bandwidth (since this design reads and writes over PCIe, a design throughput of 10.7 GB/s uses 10.7 x 2 = 21.4 GB/s of total PCIe bandwidth). However, the single-kernel and multi-kernel designs saturate the PCIe bandwidth with a latency that is ~473x lower than the offload kernel.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
