# `Optimization Report` Sample

The Optimization Report sample shows basic auto-vectorization and how to use the optimization report option to analyze your code and identify potential points of performance improvement.   

| Area                      | Description
|:---                       |:---
| What you will learn       | How to use auto-vectorization and the optimization report option with the Intel® oneAPI DPC++/C++ Compiler. 
| Time to complete          | 10 minutes

## Purpose

The Intel® oneAPI DPC++/C++ Compiler has an auto-vectorization mechanism that detects operations in the application that can be done in parallel and converts sequential operations to parallel operations by using the Single Instruction Multiple Data (SIMD) instruction set. 

For the Intel® oneAPI DPC++/C++ Compiler, vectorization is the unrolling of a loop combined with the generation of packed SIMD instructions. Because the packed instructions operate on more than one data element at a time, the loop can execute more efficiently. It is sometimes referred to as auto-vectorization to emphasize that the compiler automatically identifies and optimizes suitable loops on its own.  

Vectorization may call library routines that can result in additional performance gain on Intel microprocessors when compared to non-Intel microprocessors. The vectorization can also be affected by certain options, such as [`-m`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/m-qm-002.html) or [`-x`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/x-qx.html).  

Vectorization is enabled with optimization levels of [`O2`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/o-001.html) and higher for both Intel® microprocessors and non-Intel® microprocessors. The default optimization level for the compiler is O2.
 
Many loops are vectorized automatically, but in cases where this does not happen, you may be able to vectorize loops by making simple code modifications.

Compiling with the [`-qopt-report`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/qopt-report-qopt-report.html) option generates an optimization report which can be used to identify potential points of performance improvement. The report option provides multiple levels of report detail about the optimization transformations done during compilation.

Intel® Advisor can assist with vectorization and show optimization report messages with your source code. Refer to [Intel® Adviser](https://www.intel.com/content/www/us/en/developer/tools/oneapi/advisor.html#gs.y0wgho) for more information.

## Prerequisites

| Optimized for             | Description
|:---                       |:---
| OS                        | Linux\*
| Hardware                  | CPU
| Software                  | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

The sample makes use of the following source files: 

| File Name                 | Description
|:---                       |:---
| Driver.c                  | <TODO>
| Multiply.c                | <TODO>
| Multiply.h                | <TODO>

>**Note**: For comprehensive information about oneAPI programming, refer to the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Establish a Performance Baseline

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* 

Create a performance baseline by compiling the sources from the src directory.

1. Change to the sample directory.
2. Build the program.

   ```
   icx -O2 -std=c17 Multiply.c Driver.c -o MatVector
   ```

3. Run the program.

   ```
   MatVector
   ```

4. The program output will be similar to the following: 

   ```
   ROW:101 COL: 101 
   Execution time is 3.853 seconds 
   GigaFlops = 5.295655 
   Sum of result = 195853.999899 
   ```

Record the execution time reported in the output. This is the baseline. 

## Generate an Optimization Report

The [`-qopt-report`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/qopt-report-qopt-report.html) option enables the generation of an optimization report at compilation. In this sample, the report is used to show what loops in the code were vectorized and to explain why other loops were not vectorized. The option enables three levels of detail in the report, with `qopt-report=1` providing minimum detail, and `qopt-report=3` providing max detail. 

For the sample program:

* `qopt-report=1` (minimum) generates a report that identifies the loops in your code that were vectorized 
* `qopt-report=2` (medium) generates a report that identifies both the loops in your code that were vectorized, and the reason that other loops were not vectorized
* `qopt-report=3` (maximum) is not used

**Note**: If you use `-qopt-report` when vectorization is disabled ([`O1`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/o-001.html)), the compiler will not generate a optimization report.

### Generate a Level 1 Optimization Report

Generate a level 1 optimization report by compiling your project with the `O2` and `qopt-report=1` options. 

1. Change to the sample directory.
2. Build the program with `O2` and `qopt-report=1` options.

   ```
   icx -O2 -std=c17 -DNOFUNCCALL -qopt-report=1 Multiply.c Driver.c -o MatVector 
   ```

   [`-DNOFUNCCALL`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/d.html) is used to tell the compiler to use the inline equivalent of the `matvec` function (found in `Driver.c`).

3. Run the program.

   ```
   MatVector 
   ```

4. Record the execution time.

   ```
   ROW:101 COL: 101 
   Execution time is 3.413 seconds 
   GigaFlops = 5.977993 
   Sum of result = 195853.999899 
   ```

The reduction in time, compared to the baseline, is mostly due to auto-vectorization of the inner loop at line 145, as noted in the vectorization report `Driver.optrpt`:  
```
LOOP BEGIN at Driver.c (140, 5) 
  
    LOOP BEGIN at Driver.c (143, 9) 
        remark #25529: Dead stores eliminated in loop 
  
        LOOP BEGIN at Driver.c (145, 13) 
            remark #15300: LOOP WAS VECTORIZED 
            remark #15305: vectorization support: vector length 2 
        LOOP END 
  
        LOOP BEGIN at Driver.c (145, 13) 
        <Remainder loop for vectorization> 
        LOOP END 
    LOOP END 
LOOP END 
```
 
**Note**: Your line and column numbers may be different.  

### Generate a Level 2 Optimization Report

Now use `qopt-report=2` to generate a report with medium details. 

1. Recompile your project with `qopt-report=2`.

   ```
   icx -std=c17 -O2 -DNOFUNCCALL -qopt-report=2 Multiply.c Driver.c -o MatVector 
   ```

The resulting report includes information about which loops were vectorized and which loops were not vectorized (and why). The vectorization report `Multiply.optrpt` indicates that the loop at line 37 in `Multiply.c` did not vectorize:  

```
LOOP BEGIN at Multiply.c (37, 5) 
<Multiversioned v2> 
    remark #15319: Loop was not vectorized: novector directive used 
  
    LOOP BEGIN at Multiply.c (49, 9) 
        remark #15319: Loop was not vectorized: novector directive used 
    LOOP END 
LOOP END 
```
 
### Generate a Level 3 Optimization Report

Now use `qopt-report=3` to generate a report with maximum details. 

1. Recompile your project with `qopt-report=3`.

   ```
   icx -std=c17 -O2 -DNOFUNCCALL -qopt-report=3 Multiply.c Driver.c -o MatVector 
   ```

The resulting report includes information about **TODO** ...

## Additional Information

For additional information about vectorization with the Intel oneAPI DPC++/C++ Compiler, refer to the [Vectorization](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/vectorization.html) section of the Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference.

For details about optimization report options, refer to [Optimization Report Options](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/optimization-report-options.html).

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).