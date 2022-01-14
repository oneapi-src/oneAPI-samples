# Private Copies
This FPGA tutorial explains how to use the `private_copies` attribute to trade off the on-chip memory use and the throughput of a DPC++ FPGA program.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04*
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | The basic usage of the `private_copies` attribute <br> How the `private_copies` attribute affects the throughput and resource use of your DPC++ FPGA program <br> How to apply the `private_copies` attribute to variables or arrays in your program <br> How to identify the correct `private_copies` factor for your program
| Time to complete                  | 15 minutes



## Purpose
This tutorial demonstrates a simple example of applying the `private_copies` attribute to an array within a loop in a task kernel to trade off the on-chip memory use and throughput of the loop.

### Description of the `private_copies` Attribute
The `private_copies` attribute is a memory attribute that enables you to control the number of private copies of any variable or array declared inside a pipelined loop. These private copies allow multiple iterations of the loop to run concurrently by providing them their own private copies of arrays to operate on. The number of concurrent loop iterations is limited by the number of private copies specified by the `private_copies` attribute.

#### Example:

Kernels in this tutorial design apply `[[intel::private_copies(N)]]` to an array declared within an outer loop and used by subsequent inner loops. These inner loops perform a global memory access before storing the results. The following is an example of such a loop:

```cpp
for (size_t i = 0; i < kMaxIter; i++) {
  [[intel::private_copies(2)]] int a[kSize];
  for (size_t j = 0; j < kSize; j++) {
    a[j] = accessor_array[(i * 4 + j) % kSize] * shift;
  }
  for (size_t j = 0; j < kSize; j++)
    r += a[j];
}
```

In this example, you only need to have two private copies of array `a` in order to have 2 concurrent outer loop iterations. The `private_copies` attribute in this example forces the compiler to create two private copies of the array `a`. In general, passing the parameter `N` to the `private_copies` attribute limits the number of private copies created for array `a` to `N`, which in turn limits the concurrency of the outer loop to `N`.

### Identifying the Correct `private_copies` Factor
Generally, increasing the number of private copies of an array within a loop situated in a task kernel will increase the throughput of that loop at the cost of increased memory use. However, in most cases, there is a limit beyond which increasing the number of private copies does not have any further effect on the throughput of the loop. That limit is the maximum exploitable concurrency of the outer loop.

The correct `private_copies` factor for a given array depends on your goals for the design, the criticality of the loop in question, and its impact on your design's overall throughput. In the example above we can analytically determine what the value of `private_copies` should be by looking at the structure of the nested loops. The two nested inner loops both require memory accesses to `a`, meaning that before a second outer loop iteration can start we would have to wait for the second inner loop to finish running. Using a `private_copies` value of 2 allows for the second outer loop iteration to start as soon as the first inner loop finishes, allowing both inner loops to run in parallel.

A typical design flow may be to:
1. Analyze your code to come up with an estimate as to what number of `private_copies` should give you the desired throughput and area for your design. Alternatively, you can rely on the compilers default heuristic, setting `private_copies` to 0, to assist with this. If choosing this option please note that the compiler heuristic might sometimes be wrong, in which case you will have rely on your own estimation.
2. Observe what impact the values have on the overall throughput and memory use of your design.
3. Choose the appropriate value that allows you to achieve your desired throughput and area goals.

## Key Concepts
* The basic usage of the `private_copies` attribute
* How the `private_copies` attribute affects the throughput and resource use of your DPC++ FPGA program
* How to apply the `private_copies` attribute to variables or arrays in your program
* How to identify the correct `private_copies` factor for your program

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `private_copies` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 12h.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

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
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake .. -DFPGA_BOARD=<board-support-package>:<board-variant>
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
3. (Optional) As the FPGA hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/private_copies.fpga.tar.gz" download>here</a>.

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX), run `cmake` using the command:

   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```
    You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=<board-support-package>:<board-variant>
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
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

*Note:* The Intel® PAC with Intel Arria® 10 GX FPGA and Intel® FPGA PAC D5005 (with Intel Stratix® 10 SX) do not support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>
*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Examining the Reports
Locate `report.html` in the `private_copies_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

On the main report page, scroll down to the section titled "Estimated Resource Usage". Each kernel name ends in the `private_copies` attribute argument used for that kernel, e.g., `kernelCompute1` uses a `private_copies` attribute value of 1. You can verify that the number of RAMs used for each kernel increases with the `private_copies` value used, with the exception of `private_copies` 0. Using `private_copies` 0 instructs the compiler to choose a default value, which is often close to the value that would give you maximum throughput.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./private_copies.fpga_emu     (Linux)
     private_copies.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./private_copies.fpga         (Linux)
     ```


### Example of Output
```
PASSED_fpga_compile
Kernel time when private_copies is set to 0: 1441.45 ms
Kernel throughput when private_copies is set to 0: 0.568 GFlops
Kernel time when private_copies is set to 1: 2916.440 ms
Kernel throughput when private_copies is set to 1: 0.281 GFlops
Kernel time when private_copies is set to 2: 1458.260 ms
Kernel throughput when private_copies is set to 2: 0.562 GFlops
Kernel time when private_copies is set to 3: 1441.450 ms
Kernel throughput when private_copies is set to 3: 0.568 GFlops
Kernel time when private_copies is set to 4: 1441.448 ms
Kernel throughput when private_copies is set to 4: 0.568 GFlops
PASSED: The results are correct
```

### Discussion of Results

The stdout output shows the throughput (GFlops) for each kernel.

When run on the Intel® PAC with Intel Arria10® 10 GX FPGA hardware board, we see that the throughput of the kernel approximately doubles when going from 1 to 2 private copies for array `a`. Increasing to 3 private copies has a very small effect, and further increasing the number of private copies does not increase the throughput achieved. This means that increasing private copies beyond 2 will incur an area penalty for little or no throughput gain. As such, for this tutorial design, optimal throughput/area tradeoff is achieved when using 2 private copies. This is as expected based on an analysis of the code, as there are two inner loops which are able to run in parallel if they each have their own private copy of the array. The small jump in throughput observed from using 3 private copies instead of 2 can be attributed to the third private copy allowing subsequent outer loop iterations to launch slightly sooner then they would with 2 private copies due to internal delays in tracking the use of each private copy.

Setting the `private_copies` attribute to 0 (or equivalently omitting the attribute entirely) produced good throughput, and the reports show us that the compiler selected 3 private copies. This does produce the optimal throughput, but in this case it probably makes sense to save some area in exchange for a very small throughput loss by specifying 2 private copies instead.

When run on the FPGA emulator, the `private_copies` attribute has no effect on kernel time. You may actually notice that the emulator achieved higher throughput than the FPGA in this example. This is because this trivial example uses only a tiny fraction of the spatial compute resources available on the FPGA.
