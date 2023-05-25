# Loop `ivdep` Sample

This sample is an FPGA tutorial that demonstrates how to apply the `ivdep` attribute to a loop to aid the compiler's loop dependence analysis.

| Area                 | Description
|:--                   |:--
| What you will learn  |  Basics of loop-carried dependencies. <br> The notion of a loop-carried dependence distance. <br> What constitutes a *safe* dependence distance. <br> How to aid the compiler dependence analysis to maximize performance.
| Time to complete     | 30 minutes
| Category             | Concepts and Functionality

## Purpose

To understand and apply `ivdep` to loops in your design, you must understand the concepts of loop-carried memory dependencies. Unlike many other attributes that can improve the performance of a design, `ivdep` has functional implications. Using it incorrectly will result in undefined behavior for your design. This sample demonstrates how to use the attribute correctly.

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

- Understanding the basics of loop-carried dependencies.
- Understanding the notion of a loop-carried dependence distance.
- Determining what constitutes a *safe* dependence distance.
- Aiding the dependence analysis to maximize performance in the compiler.

### Loop-Carried Memory Dependencies

A *loop-carried memory dependency* refers to a situation where memory access in a given loop iteration cannot proceed until a memory access from a previous loop iteration is completed. Loop-carried dependencies can be categorized into the following cases:

- **True-dependence (Read-After-Write)** - A memory location read in an iteration that must occur after a previous iteration writes to the same memory location.
- **Anti-dependence (Write-After-Read)** - A memory location read must occur before a future iteration writes to the same memory location.
- **Output-dependence (Write-After-Write)** - A memory location write must occur before a future iteration writes to the same memory location.

The compiler employs static analysis to scan the program code to establish the dependence relationships between all memory accesses in a loop; however, depending on the complexity of the addressing expressions and the stride or upper bound of the loop, the compiler may not be able to determine statically precise dependence information.

In such scenarios, the compiler must conservatively assume some statements to be dependent in order to guarantee the functional correctness of the generated hardware. Precise dependence information is crucially important to generate an efficient pipelined datapath. Such information reduces the number of assumed dependencies, allowing the hardware schedule to extract as much pipeline parallelism from loops as possible.

#### Example 1: Basic True-Dependence

Each iteration of the loop reads a value from the memory location written to in the previous iteration. The pipelined datapath generated by the compiler cannot issue a new iteration until the previous iteration is complete.

```c++
for(i = 1; i < n; i++){
  S: a[i] = a[i-1];
}
```

#### Example 2: Complex or Statically-Unknown Indexing Expression

The compiler cannot statically infer the true access pattern for the loads from array `a`. To guarantee functional correctness, the compiler must conservatively assume the statements in the loop to be dependent across all iterations. The resulting generated datapath issues new iterations, similar to the example 1, executing one iteration at a time.

```c++
for(i = 0; i < n; i++){
  S: a[i] = a[b[i]];
}
```

#### Example 3: Loop-Independent Dependence

Some memory dependencies in program code do not span multiple iterations of a loop. In the following example code, dependencies from statement `S2` on `S1` and from statement `S3` on `S1` are referred to as loop-independent memory dependencies. Such dependencies do not prevent the compiler from generating an efficient pipelined loop datapath and are not considered in this tutorial.

```c++
for(i = 0; i < n; i++){
  S1: a[i] = foo();
  ...
  S2: b[i] = a[i];
}
for(j = 0; j < m; j++){
  S3: a[i] = bar();
}
```

### Loop-Carried Dependence Distance

Imagine loop-carried dependencies in terms of the distance between the dependence source and sink statements, measured in the number of iterations of the loop containing the statements. In example 1, the dependence source (store into array `a`) and dependence sink (load from the same index in array `a`) are one iteration apart. That is, for the specified memory location, the data is read one iteration after it was written. Therefore, this true dependence has a distance of 1. In many cases, the compiler loop dependence analysis may be able to statically determine the dependence distance.

#### Example 4: Simple Dependence Distance

The compiler's static analysis facilities can infer that the distance of the true dependence in the following example code is ten iterations. This impacts the scheduling of how iterations of the loop are issued into the generated pipelined datapath. For example, iteration `k` may not begin executing the load from array `a` before iteration `(k-10)` has completed storing the data into the same memory location. However, iterations `[k-9,k)` do not incur the scheduling constraint on the store in iteration `(k-10)` and begin execution earlier.

```c++
for(i = 1; i < n; i++){
  S: a[i] = a[i-10];
}
```

#### Example 5: Dependence Distance Across Multiple Loops in a Nest

In the code snippet that follows, Statement `S` forms two distinct true dependencies, one carried by loop `L1` and one by loop `L2`. Across iterations of loop `L1`, data is stored into a location in array `a` that is read in the next iteration. Similarly, across iterations of loop `L2`, data is stored into a location in array `a` that is read in a later iteration. In the latter case, the dependence across loop `L2` has dependence distance of 2. In the former, the dependence distance across loop `L1` has dependence distance of 1. Special care must be taken when reasoning about loop-carried memory dependencies spanning multiple loops.

```c++
L1: for(i = 1; i < n; i++){
  L2: for(j = 1; j < m; j++){
        S: a[i][j] = a[i-1][j-2];
  }
}
```

### Specifying That Memory Accesses Do *Not* Cause Loop-Carried Dependencies

Apply the `ivdep` attribute to a loop to inform the compiler that ***none*** of the memory accesses within a loop incur loop-carried dependencies.

```c++
[[intel::ivdep(a)]]
for (int i = 0; i < n; i++) {
    a[i] = a[i - X[i]];
}
```
The `ivdep(a)` attribute indicates to the compiler that it can disregard assumed loop-carried memory dependencies on accesses to array `a`. Disregarding dependencies on `a` allows the compiler to generate a pipelined datapath for this loop capable of issuing new iterations as soon as possible (every cycle), maximizing possible throughput.

The `ivdep` attribute can also be applied to a loop without a specific array or pointer argument. When applied in this manner, the `ivdep` attribute indicates to the compiler that it can ignore all assumed loop-carried dependencies on accesses to all arrays and pointers with the loop.

>**Important**: As a best practice, you should always apply `ivdep` with an array or pointer specified so that they explicitly understand which accesses are affected. Specifying `ivdep` incorrectly by telling the compiler to disregard loop-carried dependencies where some exist results in undefined (and likely incorrect) behavior.

### Specifying That Memory Accesses Do *Not* Cause Loop-Carried Dependencies Across a Fixed Distance

Apply the `ivdep` attribute with an additional `safelen` parameter to set a specific lower bound on the dependence distance that can possibly be attributed to loop-carried dependencies in the associated loop.

```c++
// n is a constant expression of integer type
[[intel::ivdep(a,n)]]
for (int i = 0; i < n; i++) {
    a[i] = a[i - X[i]];
}
```
The `ivdep(a,n)` attribute informs the compiler to generate a pipelined loop datapath that can issue a new iteration as soon as the iteration `n` iterations ago has completed. The attribute parameter (`safelen`) is a refinement of the compiler loop-carried dependence static analysis that infers the dependence present in the code but is otherwise unable to determine its distance accurately.

>**Important**: Applying the `ivdep` attribute or the `ivdep` attribute with a `safelen` parameter may lead to incorrect results if the annotated loop exhibits loop-carried memory dependencies. The attribute directs the compiler to generate hardware assuming no loop-carried dependencies. Specifying this assumption incorrectly is an invalid use of the attribute and results in undefined (and likely incorrect) behavior.

### Testing the Tutorial

In `loop_ivdep.cpp`, the `ivdep` attribute is applied to the kernel work loop with a `safelen` parameter of 1 and 128.

```c++
  TransposeAndFold<kMinSafelen>(selector,  a,  b); // kMinSafelen = 1
  TransposeAndFold<kMaxSafelen>(selector,  a,  b); // kMaxSafelen = 128
```
The `ivdep` attribute with the `safelen` parameter equal to 1 informs the compiler that other iterations of the associated loop do not form a loop-carried memory dependence with a distance of at least 1. That is, the attribute is redundant and is equivalent to the code without the attribute in place.

Try to compile the tutorial program in `loop_ivdep.cpp` with and without the `[[intel::ivdep]]` attribute altogether and compare the resulting reports.

The `ivdep` attribute with `safelen` parameter equal to 128 is reflective of the maximum number of iterations of the associated loop among which no loop-carried memory dependence occurs. The annotated loop nest contains a dependence on values of array `temp_buffer`:

```c++
for (size_t j = 0; j < kMatrixSize * kRowLength; j++) {
  for (size_t i = 0; i < kRowLength; i++) {
    temp_buffer[j % kRowLength][i] += in_buffer[i][j % kRowLength];
  }
}
```
Observe that the indexing expression on `temp_buffer` evaluates to the same index every `kRowLength` iterations of the `j` loop. Specifying the `ivdep` attribute on the `j` loop without a `safelen` parameter, or with a `safelen` parameter >= `kRowLength` leads to undefined behavior because the generated hardware does not adhere to the ordering constraint imposed by the dependence. Specifying the `ivdep` attribute with a `safelen` attribute <= `kRowLength` is valid and will generate better performing results.

## Build the `ivdep` Tutorial

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

Locate `report.html` in the `loop_ivdep_report.prj/reports/` directory.

Navigate to the Loops Analysis section of the optimization report and look at the initiation interval (II) achieved by the two kernel versions.

- `safelen(1)`: The II reported for this version of the kernel is five cycles.
You should see a message similar to "Compiler failed to schedule this loop with smaller II due to memory dependency."
- `safelen(128)`: The II reported for this version of the kernel is one cycle, the optimal result. You should see a message similar to  "a new iteration is issued into the pipelined loop datapath on every cycle".


## Run the `ivdep` Sample

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./loop_ivdep.fpga_emu
   ```
2. Run the sample on the FPGA simulator device.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./loop_ivdep.fpga_sim
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   ./loop_ivdep.fpga
   ```

### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   loop_ivdep.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator device.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   loop_ivdep.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
3. Run the sample on the FPGA device (only if you ran `cmake` with `-DFPGA_DEVICE=<board-support-package>:<board-variant>`).
   ```
   loop_ivdep.fpga.exe
   ```

## Example Output

```
SAFELEN: 1 -- kernel time : 50.9517 ms
Throughput for kernel with SAFELEN 1: 1286KB/s
SAFELEN: 128 -- kernel time : 10 ms
Throughput for kernel with SAFELEN 128: 6277KB/s
PASSED: The results are correct
```

The following table summarizes the execution time (in ms) and throughput (in MFlops) for `safelen` parameters of 1 (redundant attribute) and 128 (`kRowLength`) for a default input matrix size of 128 x 128 floats on Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA and the Intel® oneAPI DPC++/C++ Compiler.

|Safelen | Kernel Time (ms) | Throughput (KB/s)
|:---    |:---              |:---
|1       | 50               | 1320
|128     | 10               | 6403

With the `ivdep` attribute applied with the maximum safe `safelen` parameter, the kernel execution time is decreased by a factor of ~5.

> **Note**: This performance difference will be apparent only when running on FPGA hardware. The emulator, while useful for verifying functionality, will generally not reflect differences in performance.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).