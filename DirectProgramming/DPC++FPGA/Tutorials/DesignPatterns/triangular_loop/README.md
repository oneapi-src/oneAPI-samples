
# Triangular Loop Optimization

This FPGA tutorial demonstrates an advanced technique to improve the performance of nested triangular loops with loop-carried dependencies in single-task kernels.
 
***Documentation***: The [FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide)  provides comprehensive instructions for targeting FPGAs through DPC++. The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a resource for general target-independent DPC++ programming. 
 
| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® Programmable Acceleration Card (PAC) with Intel Stratix® 10 SX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | How and when to apply the triangular loop optimization technique
| Time to complete                  | 30 minutes
 
_Notice: Limited support in Windows*; compiling for FPGA hardware is not supported in Windows*_

## Purpose

This FPGA tutorial introduces an advanced optimization technique to improve the performance of nested triangular loops with loop-carried dependencies. Such structures are challenging to optimize because of the time-varying loop trip count.

### What is a triangular loop?

A triangular loop is a loop nest where the inner-loop range depends on the outer loop variable in such a way that the inner-loop trip-count shrinks or grows. This is best explained with an example:

```c++
  for (int x = 0; x < n; x++) {
    for (int y = x + 1; y < n; y++) {
      local_buf[y] = local_buf[y] + SomethingComplicated(local_buf[x]);
    }
  }
```

In this example, the inner-loop executes fewer and fewer iterations as overall execution progresses. Each iteration of the inner-loop performs a read from index `[x]` and a read-modify-write on indices `[y]=x+1` to `[y]=n-1`. Expressed graphically (with _n_=10), these operations look like:

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

### Performance challenge

In the above example, the table shows that in outer-loop iteration `x=0`, the program reads `local_buf[x=0]` and reads, modifies, and writes the values from `local_buf[y=1]` through `local_buf[y=9]`. This pattern of memory accesses results in a loop-carried dependency across the outer loop iterations. For example, the read at `x=2` depends on the value that was written at `x=1,y=2`. 

Generally, a new iteration is launched on every cycle as long as a sufficient number of inner-loop 
iterations are executed *between* any two iterations that are dependent on one another.

However, the challenge in the triangular loop pattern is that the trip-count of the inner-loop
progressively shrinks as `x` increments. In the worst case of `x=7`, the program writes to `local_buf[y=8]` in the first `y` iteration, but has only one intervening `y` iteration at `y=9` before the value must be read again at `x=8,y=8`. This may not allow enough time for the write operation to complete. The compiler compensates for this by increasing the initiation interval (II) of the inner-loop to allow more time to elapse between iterations. Unfortunately, this reduces the throughput of the inner-loop by a factor of II.

A key observation is that this increased II is only functionally necessary when the inner-loop trip-count becomes small. Furthermore, the II of a loop is static -- it applies for all invocations of that loop. Therefore, if the *outer-loop* trip-count (_n_) is large, then most of the invocations of the inner-loop unnecessarily suffer the aforementioned throughput degradation. The optimization technique demonstrated in this tutorial addresses this issue.

### Optimization concept

The triangular loop optimization alters the code to guarantee that the trip count never falls below some minimum (_M_). This is accomplished by executing extra 'dummy' iterations of the inner loop when the *true* trip count falls below _M_. 

The purpose of the dummy iterations is to allow extra time for the loop-carried dependency to resolve. No actual computation (or side effects) take place during these added iterations. Note that the extra iterations are only executed on inner loop invocations that require them. When the inner-loop trip count is large, extra iterations are not needed. 

This technique allows the compiler to achieve II=1. 

Applying the triangular loop optimization to the original example, the post-optimization execution graph for _M_=6 (with _n_=10) appears as follows:

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

### Selecting the value of _M_

The objective is to find the minimal value of _M_ that enables the compiler to achieve an II of 1. Any value of _M_ larger than this minimum adds unnecessary latency to the computation.

A good starting point of the value of _M_ is the II of the unoptimized inner loop, which can be found in the "Loop Analysis" report of the unoptimized code. If the compiler can achieve II=1 with this starting value, experiment with reducing _M_ until II increases. If the compiler does not achieve II=1, increase _M_ until it does. This search for the optimal _M_ can be done quickly, as the compiler takes little time to generate the static optimization report.

### Applying the optimization in code

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

***Single loop:*** Notice that the original nested loop has been manually coalesced or "merged" into a single loop. The explicit `x` and `y` induction variables are employed to achieve the triangular iteration pattern. The actual computation inside the loop is guarded by the condition `y > x`.

***Merged loop trip count:*** The total trip-count of this merged loop is `loop_bound` in the snippet . The value of `loop_bound` is the total number of iterations in the execution graph diagram, which is a function of _n_ and _M_.

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
descending series from `n` is `n*(n+1)/2`. Since there is no iteration at `x=9,y=9`, subtract 1  (i.e., `n*(n+1)/2 - 1`). When _n_=10, this expression yields 54, as expected.

The number of dummy iterations on the right is 4+3+2+1 = 10. The largest number in this series is _M_-2. Using the same formula for a descending series , you get `(M-2)*(M-1)/2`. For _M_=6, this this expression yields 4*5/2 = 10, as expected.

Summing the number of real and dummy iterations gives the total iterations of the merged loop.

***Use of ivdep***: Since the loop is restructured to ensure that a minimum of M iterations are executed, the  `[[intelfpga::ivdep(M)]]` is used to hint to the compiler that iterations with dependencies are always separated by at least M iterations.



## Key Concepts
* The triangular loop advanced optimization technique, and situations in which it is applicable
* Using `ivdep safelen` to convey the broken loop-carried dependency to the compiler

## License  
This code sample is licensed under MIT license.


## Building the `triangular_loop` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (fpga_compile or fpga_runtime) as well as whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/get-started/base-toolkit/](https://devcloud.intel.com/oneapi/get-started/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.

### On a Linux* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:  
    ```
    cmake ..
   ```
   Alternatively, to compile for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
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
3. (Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/triangular_loop.fpga.tar.gz" download>here</a>.

### On a Windows* System
Note: `cmake` is not yet supported on Windows. A build.ninja file is provided instead. 

1. Enter the source file directory.
   ```
   cd src
   ```

2. Compile the design. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device): 
      ```
      ninja fpga_emu
      ```

   * Generate the optimization report:

     ```
     ninja report
     ```
     If you are targeting Intel® PAC with Intel Stratix® 10 SX FPGA, instead use:
     ```
     ninja report_s10_pac
     ```     
   * Compiling for FPGA hardware is not yet supported on Windows.
 
 ### In Third-Party Integrated Development Environments (IDEs)
 
You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `triangular_loop_report.prj/reports/` or `triangular_loop_s10_pac_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

Consult the "Loop Analysis" report to compare the optimized and unoptimized versions of the loop.


## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./triangular_loop.fpga_emu     (Linux)
     triangular_loop.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./triangular_loop.fpga         (Linux)
     ```

### Example of Output

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

### Discussion of Results
A test compile of this tutorial design achieved an f<sub>MAX</sub> of approximately 210 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results with and without the optimization are shown in the following table:

Configuration | Overall Execution Time (ms) | Throughput (MB/s)
-|-|-
Without optimization | 4972 | 25.7
With optimization | 161 | 796.6

Without optimization, the compiler achieved an II of 30 on the inner-loop. With the optimization, the compiler achieves an II of 1 and the throughput increased by approximately 30x.

