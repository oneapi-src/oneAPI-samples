# Loop `initiation_interval` Sample

This sample is an FPGA tutorial that demonstrates how a user can use the `intel::initiation_interval` attribute to change the initiation interval (II) of a loop in scenarios that this feature improves performance.

| Area                 | Description
|:--                   |:--
| What you will learn  | The f<sub>MAX</sub>-II tradeoff <br>Default behavior of the compiler when scheduling loops. <br> How to use `intel::initiation_interval` to attempt to set the II for a loop. <br> Scenarios in which `intel::initiation_interval` can be helpful in optimizing kernel performance.
| Time to complete     | 20 minutes
| Category             | Concepts and Functionality

## Purpose

This FPGA tutorial demonstrates how to use the `intel::initiation_interval` attribute to set the II for a loop. The attribute serves two purposes:

- Relax the II of a loop with a loop-carried dependency in order to achieve a higher kernel f<sub>MAX</sub>
- Enforce the II of a loop such that the compiler will error out if it cannot achieve the specified II

>**Note**: The tutorial assumes you are familiar with the concepts of [loop-carried dependencies](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/optimize-your-design/throughput-1/single-work-item-kernels/single-work-item-kernel-design-guidelines.html#SECTION_3A389B8F1FE3452C84F44F07FA2C813E) and [initiation interval (II)](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/fpga-optimization-flags-attributes-pragmas-and-extensions/loop-directives/ii-attribute.html).

- A **loop-carried dependency** refers to a situation where an operation in a loop iteration cannot proceed until an operation from a previous loop iteration has completed.
- The **initiation interval**, or **II**, is the number of clock cycles between the launch of successive loop iterations.

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

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

The sample illustrates the following important concepts.

- The f<sub>MAX</sub>-II tradeoff.
- Default behavior of the compiler when scheduling loops.
- How to use `intel::initiation_interval`  to set the II for a loop.
- Scenarios in which `intel::initiation_interval` can be helpful in optimizing kernel performance.

The `intel::initiation_interval` attribute is useful when optimizing kernels with loop-carried dependencies in loops with a short trip count, to prevent the compiler from scheduling the loop with a f<sub>MAX</sub>-II combination that results in low system-wide f<sub>MAX</sub>, decreasing throughput.

### The f<sub>MAX</sub>-II Tradeoff

Generally, striving for the lowest possible II of 1 is preferred. However, in some cases, it may be suboptimal for the scheduler to do so.

For example, consider a loop with loop-carried dependencies. The compiler must ensure that these dependencies are satisfied. To achieve an II of 1, the compiler must schedule all of the operations necessary to compute loop-carried dependencies within a single clock cycle. As the number of operations in a clock cycle increases, the circuit's clock frequency (f<sub>MAX</sub>) must decrease. The lower clock frequency slows down the entire circuit, not just the single loop. This is the f<sub>MAX</sub>-II tradeoff. Sometimes, the benefits of achieving an II of 1 for a particular loop may not outweigh the negative impact of reducing f<sub>MAX</sub> for the entire system.

In the presence of loop-carried dependencies, it may be impossible for the compiler to schedule a given loop with II = 1 while respecting a target f<sub>MAX</sub>.

![High fMAX with II 1](assets/high_fmax_low_ii.png)

In this case, the compiler can either:

- Increase the cycle time (trading off f<sub>MAX</sub>) to allow operations with loop-carried dependencies to be executed in one clock cycle in order to achieve an II of 1.

  ![Low fMAX with II 1](assets/low_fmax_low_ii.png)

- Maintain the cycle time so the loop body executes in multiple clock cycles, while increasing the number of clock cycles between subsequent loop iterations (trading off II), until the next loop iteration is able to execute after the last operation of a loop-carried dependency has finished.

  ![High fMAX with II 3](assets/high_fmax_high_ii.png)


The `intel::initiation_interval` attribute gives the user explicit control over the f<sub>MAX</sub>-II tradeoff.

### Compiler Default Heuristics and Overrides

By default, the compiler attempts to schedule each loop with the optimal minimum product of the II and cycle time (1/f<sub>MAX</sub>), while ensuring that all loop carried dependencies are fulfilled. The resulting loop block might not necessarily achieve the targeted f<sub>MAX</sub> as the f<sub>MAX</sub>-II heuristic depends on low II or high f<sub>MAX</sub>. A combination of f<sub>MAX</sub> and II may have the best heuristic but might not necessarily achieve the target f<sub>MAX</sub>. This might cause performance bottlenecks as f<sub>MAX</sub> is a global constraint and II is a local constraint.

The `intel::initiation_interval` attribute can be used to specify an II for a particular loop. It informs the compiler to ignore the default heuristic and to try and schedule the loop that the attribute is applied to with the specific II the user provides.

The targeted f<sub>MAX</sub> can be specified using the [`–Xsclock`](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/fpga-optimization-flags-attributes-pragmas-and-extensions/optimization-flags/specify-schedule-fmax-target-for-kernels-xsclock-clock-target.html) compiler argument. The argument determines the pipelining effort of the compiler, which uses an internal model of the FPGA fabric to estimate f<sub>MAX</sub>. The true f<sub>MAX</sub> is known only after compiling to hardware. Without the argument, the default target f<sub>MAX</sub> is 240MHz for the Intel® Arria® 10 FPGAs and 480MHz for the Intel® Stratix® 10 and Agilex® 7 FPGAs, but the compiler will not strictly enforce reaching that default target when scheduling loops.

>**Note:** The scheduler prioritizes II over f<sub>MAX</sub> if both `-Xsclock` **and** `intel::initiation_interval` are used. Your kernel may be able to achieve a lower II for the loop with the `intel::initiation_interval` attribute while targeting a specific f<sub>MAX</sub>, but the loop will not be scheduled with the lower II.

### Syntax

To let the compiler attempt to set the II for a loop to a positive constant expression of integer type *n*, declare the attribute above the loop. For example:

```cpp
[[intel::initiation_interval(n)]] // n is required
for (int i = 0; i < N; i++) {
  s *= a;
  s += b;
}
```

### Use Cases for `intel::initiation_interval`

1. Allow users to assert an II for a loop.

   This is useful during development when making changes that could potentially compromise the previously achieved II. Upon finding out that a loop can be scheduled with a specific II, one can use the `intel:ii` attribute to set the achieved II as the II the compiler must achieve. If the compiler is unable to schedule the loop with the same II as before after some new changes during development, it will produce an error. This allows changes causing throughput drops to be easily identified in larger designs.

2. Alter the compiler's default f<sub>MAX</sub>-II tradeoff, usually by relaxing II.

   An in-depth example is given in this code sample.

### Code Sample: Overriding the f<sub>MAX</sub>-II Heuristic in the Compiler

The code sample gives a trivial kernel in which the choice made by the compiler is suboptimal and the `intel::initiation_interval` attribute can be used to improve performance.

This tutorial contains two distinct pipelineable loops:

- A short-running initialization loop that has a long feedback path as a result loop-carried dependence
- A long-running loop that does the bulk of the processing, with a feedback path

>**Note:** The operations performed in the short and long-running loops are for illustrative purposes only.

Since the tutorial shows performance impacts in terms of f<sub>MAX</sub> and all kernels are implemented by the compiler in a common clock domain, the results cannot be shown in two kernels that are compiled once. To see the impact of the `intel::initiation_interval` optimization in this tutorial, compile the design twice.

Part 1 compiles the kernel code without setting the `ENABLE_II` macro, whereas Part 2 compiles the kernel while setting this macro. The macro chooses between two code segments that are functionally equivalent, but the `ENABLE_II` version of the code demonstrates the two use cases of `intel::initiation_interval`.

#### Part 1: Without `ENABLE_II`

According to the default behavior, the compiler does not know that the initialization loop has a smaller impact on the overall throughput. Thus, the compiler schedules the loop using the minimum II/f<sub>MAX</sub> ratio. Because the initialization loop has a loop-carried dependence, it has a feedback path in the generated hardware. The targeted clock frequency might not be achieved by the scheduler when optimizing for the minimum II/f<sub>MAX</sub>.

Depending on the feedback path in the long-running loop, the rest of the kernel could have run at a higher f<sub>MAX</sub>, which is the case in this design. The long-running loop is able to achieve an II of 1 while targeting the default f<sub>MAX</sub> but will be bottlenecked by the highest f<sub>MAX</sub> achieved by all blocks, resulting in lowered throughput.

#### Part 2: With `ENABLE_II`

In this part, `intel::initiation_interval` is used for both the short and long running loops to show the two scenarios where using the attribute is appropriate.

The first `intel::initiation_interval` declaration sets an II value of 3 for the Intel® Arria® 10 FPGA, and an II value of 5 for the Intel® Stratix® 10 and Agilex® 7 FPGAs. Since the initialization loop has a low trip count compared to the long-running loop, a higher II for the initialization loop is a reasonable tradeoff to allow for a higher overall f<sub>MAX</sub> for the entire kernel.

>**Note:** For Intel® Stratix® 10 FPGA, the estimated f<sub>MAX</sub> of the long-running loop is not able to reach the default targeted f<sub>MAX</sub> of 480MHz while maintaining an II of 1. This is due to the nature of the feedback path that exists in the long running loop. Setting the II of the initialization loop to 5 ensures that the initialization loop is not the bottleneck when finding the maximum operating frequency.

The second `intel::initiation_interval` declaration sets an II of 1 for the long-running loop. We might not want to compromise the II of 1 achieved for this loop while performing optimizations on other parts of the kernel. By declaring that the loop should have an II of 1, the compiler will produce an error if it cannot schedule this loop with that II. The error implies that the other optimization will have a negative performance impact on this loop. This makes it easier to find the cause of any throughput drops in larger designs.

## Build the `Loop Initiation Interval` Tutorial

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
   >  cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   >  ```
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
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

### Read the Reports

Locate the pair of `report.html` files in either:

- **Report-only compile**:  `loop_ii_report.prj` and `loop_ii_enable_ii_report.prj`
- **FPGA hardware compile**: `loop_ii.prj` and `loop_ii_enable_ii.prj`

Looking at the reports for the design without the `intel::initiation_interval` attribute, navigate to the *Loop Analysis* report (*Throughput Analysis* > *Loop Analysis*). Click the *SimpleMath* kernel in the *Loop List* panel and use the *Bottlenecks* viewer panel in the bottom left. You will see that a throughput bottleneck exists in the *SimpleMath* kernel.

Select the bottleneck. The report shows that the estimated f<sub>MAX</sub> is significantly lower than the target f<sub>MAX</sub> and shows the feedback path responsible, which is the feedback path in the initialization loop.

The *Loop Analysis* report shows that the long-running loop achieves the target f<sub>MAX</sub> with an II of 1.

Compare the results to the report for the version of the design using the `intel::initiation_interval` attribute. Here both loops achieve the target f<sub>MAX</sub>.

> **Note**: Only the report generated after the FPGA hardware compile will reflect the true performance benefit of using the `initiation_interval` extension. The difference is **not** apparent in the reports generated by `make report` because a design's f<sub>MAX</sub> cannot be predicted. The final achieved f<sub>MAX</sub> can be found in `loop_ii.prj/reports/report.html` and `loop_ii_enable_ii.prj/reports/report.html` (after `make fpga` completes), in *Clock Frequency Summary* on the main page of the report.

## Run the `Loop Initiation Interval` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./loop_ii.fpga_emu
   ```

2. Run the sample on the FPGA simulator device (the kernel executes on the CPU).
   ```
   # Sample without intel::initiation_interval attribute
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./loop_ii.fpga_sim
   # Sample with intel::initiation_interval attribute
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./loop_ii_enable_ii.fpga_sim
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   # Sample without intel::initiation_interval attribute
   ./loop_ii.fpga
   # Sample with intel::initiation_interval attribute
   ./loop_ii_enable_ii.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   loop_ii.fpga_emu.exe
   ```

2. Run the sample on the FPGA simulator device (the kernel executes on the CPU).
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   # Sample without intel::initiation_interval attribute
   loop_ii.fpga_sim.exe
   loop_ii_enable_ii.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```

3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   # Sample without intel::initiation_interval attribute
   loop_ii.fpga.exe
   # Sample with intel::initiation_interval attribute
   loop_ii_enable_ii.fpga.exe
   ```

## Example Output

Output of sample without the `intel::initiation_interval` attribute.

```
Kernel Throughput: 0.0635456MB/s
Exec Time: 60.0309s , InputMB: 3.8147MB
PASSED
```

Output of sample with the `intel::initiation_interval` attribute.

```
Kernel_ENABLE_II Throughput: 0.117578MB/s
Exec Time: 32.4439s , InputMB: 3.8147MB
PASSED
```

Total throughput improved with the use of the `intel::initiation_interval` attribute because the increase in kernel f<sub>MAX</sub> is more significant than the II relaxation of the low trip-count loop.

This performance difference will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).