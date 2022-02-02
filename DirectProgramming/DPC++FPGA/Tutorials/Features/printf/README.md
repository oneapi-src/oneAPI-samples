
# Print Data Using SYCL Printf
This FPGA tutorial explains how to use the `sycl::ext::oneapi::experimental::printf` to print in a DPC++ FPGA program.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA <br> Intel® FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix® 10 SX) <br> Intel® FPGA 3rd party / custom platforms with oneAPI support <br> *__Note__: Intel® FPGA PAC hardware is only compatible with Ubuntu 18.04* 
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit 
| What you will learn               | How to declare and use printf in a DPC++ program
| Time to complete                  | 10 minutes


## Purpose
This tutorial shows how to use some simple macros to enable easy use of the SYCL printf() function in a DPC++ FPGA program. This function allows printing from within code running on the FPGA.

### Motivation
Previously, we've provided examples for how to print data in DPC++ using the Stream class in the [Intel oneAPI GPU Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/debugging-and-profiling/doing-io-in-the-kernel.html).

Compare to the Stream class, printf() has the following advantages:

* printf() is a function that is globally available; Stream is an object which can only be obtained by passing it as a kernel argument from the host. In order to use Stream somewhere within your application, you have to pass the Stream object through the whole call stack. For debugging in large applications, using printf() has a great advantage allowing you to quickly add some print statements and easy to remove later without changing your main code.

* On the FPGA device, printf() has smaller area usage (less LSUs) and better performance (Stream could introduce massive II inner loops).

### Simple Code Example

Using printf can be verbose in DPC++ kernels. To simplify, add the following macro to your code.

```cpp
#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif

#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

PRINTF("Hello, World!\n");
PRINTF("Hello: %d\n", 123);
```

## Key Concepts

* How to use `printf`
* Advantages of `printf`
    * Easy to use
    * Smaller area usage and better performance
* [Limitations](#known-issues-and-limitations) of `printf`

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `printf` Tutorial

### Include Files
The included header `dpc_common.hpp` is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

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
3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/printf.fpga.tar.gz" download>here</a>.

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
Locate `report.html` in the `printf_report.prj/reports/` directory. Open the report in any of Chrome*, Firefox*, Edge*, or Internet Explorer*.

From the report, you can find the compilation information of the design and the estimated FPGA resource usage. For example, navigate to the Area Analysis section of the report (Kernel System -> BasicKernel->Computaion), you can see the estimated resource usage(ALUTs, FFs, RAMs, etc.) for each printf() call.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./printf.fpga_emu     (Linux)
     printf.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA device:
     ```
     ./printf.fpga         (Linux)
     ```

### Example of Output
```
Result1: Hello, World!
Result2: %
Result3: 123
Result4: 123
Result5: 1.00
Result6: print slash_n \n 
Result7: Long: 650000
Result8: Preceding with blanks:       1977 
Result9: Preceding with zeros: 0000001977 
Result10: Some different radices: 100 64 144 0x64 0144 
Result11: ABCD
```

## Known issues and limitations

There are some known issues with the experimental::printf and that's why the function is in the experimental namespace. The following limitations exist when using experimental::printf() on FPGA hardware:

* Printing string literals %s is not supported yet. You can put the string directly in the format as a workaround. For example: `PRINTF("Hello, World!\n")`
* Printing pointer address %p is not supported yet.
* If you have multiple PRINTF statements in the kernel, the order of printed data in the stdout might not obey the sequential order of those statements in the code.
* Buffer is only flushed to stdout after the kernel finishes in hardware.

