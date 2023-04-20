
# `Triangular Loop` Sample

This FPGA sample demonstrates an advanced technique to improve the performance of nested triangular loops with loop-carried dependencies in single-task kernels.

| Area                 | Description
|:--                   |:--
| What you will learn  | How and when to apply the triangular loop optimization technique
| Time to complete     | 30 minutes
| Category             | Code Optimization

## Purpose

This FPGA sample introduces an advanced optimization technique to improve the performance of nested triangular loops with loop-carried dependencies. Such structures are challenging to optimize because of the time-varying loop trip count.

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

The sample illustrates the following important concepts:

- The triangular loop advanced optimization technique and situations where it works best.
- The use of `ivdep safelen` to convey the broken loop-carried dependency to the compiler.

### Triangular Loops Explained

A triangular loop is a loop nest where the inner-loop range depends on the outer loop variable in such a way that the inner-loop trip-count shrinks or grows. This is best explained with an example:

```c++
  for (int x = 0; x < n; x++) {
    for (int y = x + 1; y < n; y++) {
      local_buf[y] = local_buf[y] + SomethingComplicated(local_buf[x]);
    }
  }
```

In this example, the inner-loop executes fewer and fewer iterations as overall execution progresses. Each iteration of the inner-loop performs a read from index `[x]` and a read-modify-write on indices `[y]=x+1` to `[y]=n-1`. Expressed graphically (with *n*=10), these operations look like:

```c++
    y=0 1 2 3 4 5 6 7 8 9
==========================
x=0   o x x x x x x x x x
x=1     o x x x x x x x x
x=2       o x x x x x x x
x=3         o x x x x x x
x=4           o x x x x x
x=5             o x x x x
x=6               o x x x
x=7                 o x x
x=8                   o x
x=9

Legend: read="o", read-modify-write="x"
```

The picture is triangular in shape, hence the name "triangular loop".

### Performance Challenge

In the above example, the table shows that in outer-loop iteration `x=0`, the program reads `local_buf[x=0]` and reads, modifies, and writes the values from `local_buf[y=1]` through `local_buf[y=9]`. This pattern of memory accesses results in a loop-carried dependency across the outer loop iterations. For example, the read at `x=2` depends on the value written at `x=1,y=2`.

Generally, a new iteration is launched on every cycle as long as a sufficient number of inner-loop
iterations are executed *between* any two iterations that are dependent on one another.

However, the challenge in the triangular loop pattern is that the trip-count of the inner-loop
progressively shrinks as `x` increments. In the worst case of `x=7`, the program writes to `local_buf[y=8]` in the first `y` iteration but has only one intervening `y` iteration at `y=9` before the value must be reread at `x=8,y=8`. This may not allow enough time for the write operation to complete. The compiler compensates for this by increasing the initiation interval (II) of the inner-loop to allow more time to elapse between iterations. Unfortunately, this reduces the throughput of the inner-loop by a factor of II.

A key observation is that this increased II is only functionally necessary when the inner-loop trip-count becomes small. Furthermore, the II of a loop is static: it applies to all invocations of that loop. Therefore, if the *outer-loop* trip-count (*n*) is large, most of the inner-loop invocations unnecessarily suffer the previously mentioned throughput degradation. The optimization technique demonstrated in this tutorial addresses this issue.

### Optimization Concept

The triangular loop optimization alters the code to guarantee that the trip count never falls below some minimum (*M*). This is accomplished by executing extra 'dummy' iterations of the inner loop when the *true* trip count falls below *M*.

The purpose of the dummy iterations is to allow extra time for the loop-carried dependency to resolve. No actual computation (or side effects) takes place during these added iterations. Note that the extra iterations are only executed on inner loop invocations that require them. When the inner-loop trip count is large, extra iterations are not needed.

This technique allows the compiler to achieve II=1.

Applying the triangular loop optimization to the original example, the post-optimization execution graph for *M*=6 (with *n*=10) appears as follows:

```c++
    y=0 1 2 3 4 5 6 7 8 9
==========================
x=0   o x x x x x x x x x
x=1     o x x x x x x x x
x=2       o x x x x x x x
x=3         o x x x x x x
x=4           o x x x x x
x=5           - o x x x x
x=6           - - o x x x
x=7           - - - o x x
x=8           - - - - o x
x=9
              <---M=6--->

Legend: read="o", read-modify-write="x", dummy iteration="-"
```

### Selecting the Value of *M*

The objective is to find the minimal value of *M* that enables the compiler to achieve an II of 1. Any value of *M* larger than this minimum adds unnecessary latency to the computation.

A good starting point of the value of *M* is the II of the unoptimized inner loop, which can be found in the "Loop Analysis" report of the unoptimized code. If the compiler can achieve II=1 with this starting value, experiment with reducing *M* until II increases. If the compiler does not achieve II=1, increase *M* until it does. This search for the optimal *M* can be done quickly, as the compiler takes little time to generate the static optimization report.

### Applying the Optimization in Code

Here is the triangular loop optimization of the original code snippet:

```c++
// Indices to track the execution in the merged loop
int x = 0, y = 1;

// Total iterations of the merged loop
const int loop_bound = TotalIterations(M, n);

[[intelfpga::ivdep(M)]]
for (int i = 0; i < loop_bound; i++) {

  // Determine if this is a real or dummy iteration
  bool compute = y > x;
  if (compute) {
    local_buf[y] = local_buf[y] + SomethingComplicated(local_buf[x]);
  }

  y++;
  if (y == n) {
    x++;
    y = Min(n - M, x + 1);
  }
}
```
This requires some explanation!

**Single loop:** Notice that the original nested loop has been manually coalesced or "merged" into a single loop. The explicit `x` and `y` induction variables are employed to achieve the triangular iteration pattern. The actual computation inside the loop is guarded by the condition `y > x`.

**Merged loop trip count:** The total trip-count of this merged loop is `loop_bound` in the snippet . The value of `loop_bound` is the total number of iterations in the execution graph diagram, which is a function of *n* and *M*.

To derive the expression for `TotalIterations(M, n)`, consider the iterations as consisting of the following two triangles of "real" and "dummy" iterations.

```c++
    y=0 1 2 3 4 5 6 7 8 9                     y=0 1 2 3 4 5 6 7 8 9
=========================                 =========================
x=0   o x x x x x x x x x                 x=0
x=1     o x x x x x x x x                 x=1
x=2       o x x x x x x x                 x=2
x=3         o x x x x x x                 x=3
x=4           o x x x x x       +         x=4
x=5             o x x x x                 x=5           -
x=6               o x x x                 x=6           - -
x=7                 o x x                 x=7           - - -
x=8                   o x                 x=8           - - - -
x=9
                                                        <(M-2)>
                                                        <---M=6--->
```
The number of "real" iterations on the left is 10+9+8+7+6+5+4+3+2 = 54. The formula for a
descending series from `n` is `n*(n+1)/2`. Since there is no iteration at `x=9,y=9`, subtract 1  (for example, `n*(n+1)/2 - 1`). When *n*=10, the expression will yield 54 as expected.

The number of dummy iterations on the right is 4+3+2+1 = 10. The largest number in this series is *M*-2. Using the same formula for a descending series , you get `(M-2)*(M-1)/2`. For *M*=6, this expression will yield 4*5/2 = 10 as expected.

Summing the number of real and dummy iterations gives the total iterations of the merged loop.

**Use of ivdep**: Since the loop is restructured to ensure that a minimum of M iterations is executed, the `[[intelfpga::ivdep(M)]]` is used to hint to the compiler that at least *M* iterations always separate any pair of dependent iterations.

## Build the `Triangular Loop` Sample

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

    1. Compile for emulation (fast compile time, targets emulated FPGA device):
       ```
       make fpga_emu
       ```
    2. Generate the optimization report:
       ```
       make report
       ```
      The report resides at `triangular_loop_report.prj/reports/report.html`.

      Consult the "Loop Analysis" report to compare the optimized and unoptimized versions of the loop.

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
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
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
      The report resides at `triangular_loop_report.prj.a/reports/report.html`.

      Consult the "Loop Analysis" report to compare the optimized and unoptimized versions of the loop.

   3. Compile for simulation (fast compile time, targets simulated FPGA device, reduced data size):
      ```
      nmake fpga_sim
      ```
   4. Compile for FPGA hardware (longer compile time, targets FPGA device):
      ```
      nmake fpga
      ```

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Triangular Loop` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./triangular_loop.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./triangular_loop.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./triangular_loop.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   triangular_loop.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   triangular_loop.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   triangular_loop.fpga.exe
   ```

## Example Output

```
Platform name: Intel(R) FPGA SDK for OpenCL(TM)
Device name: pac_a10 : Intel PAC Platform (pac_ec00000)

Length of input array: 8192

Beginning run without triangular loop optimization.

Verification PASSED

Execution time: 4.240185 seconds
Throughput without optimization: 30.187364 MB/s

Beginning run with triangular loop optimization.

Verification PASSED

Execution time: 0.141516 seconds
Throughput with optimization: 904.489876 MB/s
```

A test compile of this tutorial design achieved an f<sub>MAX</sub> of approximately 210 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results with and without the optimization are shown in the following table:

Configuration         | Overall Execution Time (ms) | Throughput (MB/s)
|:---                 |:---                         |:---
|Without optimization | 4972                        | 25.7
|With optimization    | 161                         | 796.6

Without optimization, the compiler achieved an II of 30 on the inner-loop. With the optimization, the compiler achieves an II of 1, and the throughput increased by approximately 30x.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
