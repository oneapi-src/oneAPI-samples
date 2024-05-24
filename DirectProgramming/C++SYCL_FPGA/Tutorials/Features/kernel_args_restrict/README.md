# `kernel_args_restrict` Sample

This sample is an FPGA tutorial that explains the  `kernel_args_restrict` attribute and the effect of the attribute on the performance of FPGA kernels.

| Area                 | Description
|:--                   |:--
| What you will learn  |  The problem of **pointer aliasing**, its impact on compiler optimizations, and how to avoid pointer aliasing. <br> The behavior of the `kernel_args_restrict` attribute and when to use it on your kernel. <br> The effect this attribute can have on your kernel's performance on FPGA.
| Time to complete     | 20 minutes
| Category             | Concepts and Functionality

## Purpose

Due to pointer aliasing, the compiler must be conservative about optimizations that reorder, parallelize or overlap operations that could alias. This tutorial demonstrates the use of the SYCL*-compliant `[[intel::kernel_args_restrict]]` kernel attribute, which should be applied anytime you can guarantee that kernel arguments do not alias. This attribute enables more aggressive compiler optimizations and often improves kernel performance on FPGA.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10 <br> Windows Server* 2019
| Hardware             | Intel® Agilex® 7, Agilex® 5, Arria® 10, Stratix® 10, and Cyclone® V FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ oneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition (or Standard Edition when targeting Cyclone® V) must be installed and accessible through your PATH.

> **Warning**: Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples.
It is categorized as a Tier 2 sample that demonstrates a compiler feature.

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
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.

## Key Implementation Details

The sample illustrates some important concepts.

- The problem of *pointer aliasing* and its impact on compiler optimizations.
- How to use the `kernel_args_restrict` attribute on your kernel (both lambda and functor coding styles).
- The effect this attribute can have on your kernel's performance on FPGA.

### Syntax

Below, we illustrate two distinct coding styles to utilize the kernel_args_restrict attribute in your SYCL kernels:

- Lambda Style

   In the lambda coding style, the attribute is applied directly within the lambda expression that defines the kernel:

   ```c++
   q.sumbit([&](handler &h) {
      // -------------------------------------------
      //          Kernel interface definition.
      // -------------------------------------------

      h.single_task<class IDLambdaKernel>([=
      ]() [[intel::kernel_args_restrict]] {
         // ----------------------------------------
         //       Kernel code implementation.
         // ----------------------------------------
      });
   });
   ```
   Here, `IDLambdaKernel` is a unique kernel ID and `q` represents a SYCL queue. Replace the comment with the actual code for your kernel.

- Functor Style
   
   Alternatively, when using the functor coding style, the attribute is applied to the call operator of a functor class that defines the kernel:

   ```c++
   struct FunctorKernel {
      // -------------------------------------------
      //         Kernel interface definition.
      // -------------------------------------------
      
      [[intel::kernel_args_restrict]]
      void operator()() const {
         // ----------------------------------------
         //       Kernel code implementation.
         // ----------------------------------------
      }
   };
   ```
   In this case, an instance of `FunctorKernel` will be invoked as a kernel by passing it to the `single_task()` function. Replace the comment with your kernel's actual implementation code.

### Pointer Aliasing Explained

Pointer aliasing occurs when the same memory location can be accessed using different *names* (i.e., variables). For example, consider the code below. Here, the value of the variable `pi` can be changed in three ways: `pi=3.14159`, `*a=3.14159` or `*b=3.14159`. In general, the compiler has to be conservative about which accesses may alias to each other and avoid making optimizations that reorder and/or parallelize operations.

```c++
float pi = 3.14;
float *a = &pi;
float *b = a;
```

### Pointer Aliasing of Arguments

Consider the function illustrated below. Though the intention of the code is clear to the reader, the compiler cannot guarantee that `in` does not alias with `out`. Imagine a degenerate case where the function was called: like this `myCopy(ptr, ptr+1, 10)`. This would cause `in[i]` and `out[i+1]` to alias to the same address, for all `i` from 0 to 9.

```c++
void myCopy(int *in, int *out, size_t int size) {
  for(size_t int i = 0; i < size; i++) {
    out[i] = in[i];
  }
}
```

This possibility of aliasing forces the compiler to be conservative. Without more information from the developer, it cannot make any optimizations that overlap, vectorize, or reorder the assignment operations. Doing so would result in functionally incorrect behavior if the compiled function is called with aliasing pointers.

If this code is compiled to FPGA, the performance penalty of this conservatism is severe. The loop in `myCopy` cannot be pipelined, because the next iteration of the loop cannot begin until the current iteration has completed.

### A Promise to the Compiler

The developer often knows that pointer arguments will never alias in practice, as with the `myCopy` function. In your program, you can use the `[[intel::kernel_args_restrict]]` attribute to inform the compiler that none of a kernel's arguments will alias to any another, which enables more aggressive optimizations. If the non-aliasing assumption is violated at runtime, the result will be undefined behavior.

C and OpenCL programmers may recognize this concept as the `restrict` keyword.

### Tutorial Code Description

In this tutorial, we will show how to use the `kernel_args_restrict` attribute for your kernel and its effect on performance. We show four kernels that perform the same function. Two kernels are designed in the lambda coding style, and the other two in functor coding style. Within each coding style category, one kernel applies the `[[intel::kernel_args_restrict]]`, while the other does not. The function of the kernel is simple: copy the contents of one buffer to another. We will analyze the effect of the `[[intel::kernel_args_restrict]]` attribute on the kernel's performance by analyzing loop II in the reports and the latency of the kernel on actual hardware.

## Build the `Kernel Args Restrict` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
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

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulates an FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the HTML optimization reports. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      make report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile and run on FPGA hardware (longer compile time, targets an FPGA device).
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
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
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
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      nmake report
      ```
   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced problem size).
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your 'build' directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory, for example:
>
>  ```
  > C:\samples\build> cmake -G "NMake Makefiles" C:\long\path\to\code\sample\CMakeLists.txt
>  ```
### Read the Reports

Locate `report.html` in the `kernel_args_restrict.report.prj/reports/` directory.

Navigate to the *Loop Analysis* report (*Throughput Analysis* > *Loop Analysis*). In the *Loop List pane*, you should see four kernels as described in the [Tutorial Code Description](#tutorial-code-description) section of this README. Each kernel has a single for-loop, which appears in the *Loop List* pane. Click the loop under each kernel to see how the compiler optimized it.

![Kernel II Overview](./assets/Kernel%20II%20Overview.png)

Compare the loop initiation interval (II) between the kernels with or without `[[intel::kernel_args_restrict]]` attribute. Notice that the loops in the *IDConservative_Lambda* and *IDConservative_Functor* kernels have large scheduled IIs, while the loops in the *IDKernelArgsRestrict_Lambda* and *IDKernelArgsRestrict_Functor* kernels have a scheduled II of 1. These IIs are estimates because the latency of global memory accesses varies with runtime conditions.

For the *IDConservative_Lambda* and *IDConservative_Functor* kernels, the compiler assumed that the kernel arguments could alias with each other. Since`out[i]` and `in[i+1]` could be the same memory location, the compiler cannot overlap the iteration of the loop performing `out[i] = in[i]` with the next iteration of the loop performing `out[i+1] = in[i+1]` (and likewise for iterations `in[i+2]`, `in[i+3]`, ...). This results in an II equal to the latency of the global memory read of `in[i]` plus the latency of the global memory write to `out[i]`.

We can confirm this by looking at the details of the loop. Click the *IDConservative_Lambda* kernel (similarly for the *IDConservative_Functor* kernel) in the *Loop List* pane and then click the loop in the *Loop Analysis* pane. Now consider the *Details* pane below. You should see something like:

![Memory Dependency Details](./assets/Memory%20Dependency%20Details.png)

The third bullet (and its sub-bullets) tells you that a memory dependency exists between the load and store operations in the loop. This is the conservative pointer aliasing memory dependency described earlier. The second bullet shows you the estimated latencies for the load and store operations (note that these are board-dependent). The sum of these two latencies (plus 1) is the II of the loop.

Next, look at the loop details of the *IDKernelArgsRestrict_Lambda* kernel (similarly for the *IDKernelArgsRestrict_Functor* kernel). You will notice that the *Details* pane doesn't show a memory dependency. The usage of the `[[intel::kernel_args_restrict]]` attribute allowed the compiler to schedule a new iteration of the for-loop every cycle since it knows that accesses to `in` and `out` will never alias.

## Run the `Kernel Args Restrict` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./kernel_args_restrict.fpga_emu
   ```

2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./kernel_args_restrict.fpga_sim
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./kernel_args_restrict.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   kernel_args_restrict.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   kernel_args_restrict.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   kernel_args_restrict.fpga.exe
   ```

## Example Output

```
Running on device: ofs_n6001 : Intel OFS Platform (ofs_ee00000)
Size of vector: 5000000 elements
Lambda kernel throughput without attribute: 5.35162 MB/s
Lambda kernel throughput with attribute: 2088.2 MB/s
Functor kernel throughput without attribute: 5.35199 MB/s
Functor kernel throughput with attribute: 2087.85 MB/s
PASSED
```

### Results Explained

The throughput observed when running the kernels with and without the `kernel_args_restrict` attribute should reflect the difference in loop II seen in the reports. The ratios will not exactly match because the loop IIs are estimates. An example ratio (compiled and run on the Intel® FPGA SmartNIC N6001-PL) is shown.

|Attribute used?  | II    | Kernel Throughput (MB/s)
|:---             |:---   |:---
|No               | ~854  | 5.35
|Yes              | ~1    | 2088

> **Note**: This performance difference will be apparent only when running on FPGA hardware. The emulator and simulator, while useful for verifying functionality, will generally not reflect differences in performance of the memory system.

## License

Code samples are licensed under the MIT license. See [License.txt](/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).
