# `Matrix Multiply` Sample

The `Matrix Multiply` sample shows how auto-vectorization can improve the performance of the sample matrix multiplication application. An optimization report is used to identify potential points of performance improvement.

| Area                      | Description
|:---                       |:---
| What you will learn       | How to use auto-vectorization and the optimization report option with the Intel® oneAPI DPC++/C++ Compiler. 
| Time to complete          | 10 minutes

## Purpose

The `Matrix Multiply` sample shows how compiler auto-vectorization can improve the performance of a program. The optimization report option is used to identify potential points of performance improvement.

The Intel® oneAPI DPC++/C++ Compiler has an auto-vectorization mechanism that detects operations in the application that can be done in parallel and converts sequential operations to parallel operations by using the Single Instruction Multiple Data (SIMD) instruction set. 

For the Intel® oneAPI DPC++/C++ Compiler, vectorization is the unrolling of a loop combined with the generation of packed SIMD instructions. Because the packed instructions operate on more than one data element at a time, the loop can execute more efficiently. It is sometimes referred to as auto-vectorization to emphasize that the compiler automatically identifies and optimizes suitable loops on its own.  

Vectorization may call library routines that can result in additional performance gains on Intel microprocessors when compared to non-Intel microprocessors. The vectorization can also be affected by certain compiler options, such as [`-m`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/m-qm-002.html) or [`-x`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/x-qx.html).  

Vectorization is enabled when optimization levels are set to [`O2`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/o-001.html) and higher for both Intel® microprocessors and non-Intel® microprocessors. The default optimization level for the compiler is `O2`.
 
Many loops are vectorized automatically, but in cases where this does not happen, you may be able to vectorize loops by making simple code modifications. Compiling with the [`-qopt-report`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/qopt-report-qopt-report.html) option generates an optimization report which can be used to identify potential points of performance improvement in the code. The report option provides multiple levels of report detail about the optimization transformations done during compilation.

Intel® Advisor can also assist with vectorization and show optimization report messages with your source code. Refer to [Intel® Adviser](https://www.intel.com/content/www/us/en/developer/tools/oneapi/advisor.html#gs.y0wgho) for more information.

## Prerequisites

| Optimized for             | Description
|:---                       |:---
| OS                        | Linux\*
| Hardware                  | CPU
| Software                  | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

The sample makes use of the following source files: 

* `driver.c`: the main program to run the matrix multiplication program and print out the time spent.
* `multiply.c`: the matrix multiplication program.
* `multiply.h`: the header file used by `multiply.c`.

The `main()` function in `driver.c` contains two possible implementations of the matrix multiplication. The first implementation is an inline execution of a double loop, gated by the `NOFUNCCALL` macro. The second implementation calls the `matvec` function, located in `multiply.c`, which contains the same double loop:

```
#ifdef NOFUNCCALL
  int i, j;
  for (i = 0; i < size1; i++) {
    b[i] = 0;
    for (j = 0;j < size2; j++) {
      b[i] += a[i][j] * x[j];
    }
  }
#else
  matvec(size1,size2,a,b,x);
#endif
```

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

Create a performance baseline by compiling and running the sample. 

1. Change to the sample directory.
2. Build the program. This will use the `matvec` function call implementation of matrix multiplication.

   ```
   icx multiply.c driver.c -o MatVector
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

Record the execution time reported in the output. This is the baseline without auto-vectorization. 

## Generate an Optimization Report

The [`-qopt-report`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/qopt-report-qopt-report.html) option enables the generation of an optimization report at compilation. In this sample, the report is used to show what loops in the code were vectorized and to explain why other loops were not vectorized. The option enables three levels of detail in the report, with `qopt-report=1` providing minimum detail, and `qopt-report=3` providing maximum detail. 

* `qopt-report=1` (minimum) generates a report that identifies the loops in your code that were vectorized.
* `qopt-report=2` (medium) generates a report that identifies both the loops in your code that were vectorized, and the reason that other loops were not vectorized.
* `qopt-report=3` (maximum) generates a report with maximum detail, including loop cost summary.

**Note**: If you use `-qopt-report` when vectorization is disabled ([`O1`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/o-001.html)), the compiler will not generate a optimization report.

### Generate a Level 1 Optimization Report

Generate a level 1 optimization report and compile to use the inline implementation of matrix multiplication.

1. Change to the sample directory.
2. Build the program with the `qopt-report=1` option to generate the report and the [`D`](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/d.html) option to specify to use the `NOFUNCCALL` implementation of matrix multiplication.

   ```
   icx -DNOFUNCCALL -qopt-report=1 multiply.c driver.c -o vec_report1 
   ```

3. Run the program.

   ```
   vec_report1 
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
LOOP BEGIN at driver.c (140, 5) 
  
    LOOP BEGIN at driver.c (143, 9) 
        remark #25529: Dead stores eliminated in loop 
  
        LOOP BEGIN at driver.c (145, 13) 
            remark #15300: LOOP WAS VECTORIZED 
            remark #15305: vectorization support: vector length 2 
        LOOP END 
  
        LOOP BEGIN at driver.c (145, 13) 
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
   icx -DNOFUNCCALL -qopt-report=2 multiply.c driver.c -o vec_report2 
   ```

The resulting report includes information about which loops were vectorized and which loops were not vectorized (and why). The optimization report `Driver.optrpt` indicates that the loop at line 119 in `driver.c` did not vectorize:

```
LOOP BEGIN at driver.c (119, 5) 
    remark #15553: loop was not vectorized: outer loop is not an auto-vectorization candidate. 
  
    LOOP BEGIN at driver.c (122, 9) 
        remark #25529: Dead stores eliminated in loop 
        remark #15553: loop was not vectorized: outer loop is not an auto-vectorization candidate. 
  
        LOOP BEGIN at driver.c (124, 13) 
            remark #15300: LOOP WAS VECTORIZED 
            remark #15305: vectorization support: vector length 2 
        LOOP END 
  
        LOOP BEGIN at driver.c (124, 13) 
        <Remainder loop for vectorization> 
        LOOP END 
    LOOP END 
LOOP END 
```

### Generate a Level 3 Optimization Report

Now use `qopt-report=3` to generate a report with maximum details. 

1. Recompile your project with `qopt-report=3`.

   ```
   icx -DNOFUNCCALL -qopt-report=3 multiply.c driver.c -o vec_report3 
   ```

In addition to information about which loops were vectorized and which were not vectorized, the level 3 report includes information about the cost of performing loops. The optimization report `Driver.optrpt` displays the loop cost summary: 

```
LOOP BEGIN at driver.c (102, 13) 
   remark #15300: LOOP WAS VECTORIZED 
   remark #15305: vectorization support: vector length 2 
   remark #15475: --- begin vector loop cost summary --- 
   remark #15476: scalar cost: 8.000000 
   remark #15477: vector cost: 7.500000 
   remark #15478: estimated potential speedup: 1.046875 
   remark #15309: vectorization support: normalized vectorization overhead 0.390625 
   remark #15570: using scalar loop trip count: 101 
   remark #15482: vectorized math library calls: 0 
   remark #15484: vector function calls: 0 
   remark #15485: serialized function calls: 0 
   remark #15488: --- end vector loop cost summary --- 
   remark #15447: --- begin vector loop memory reference summary --- 
   remark #15450: unmasked unaligned unit stride loads: 2 
   remark #15451: unmasked unaligned unit stride stores: 0 
   remark #15456: masked unaligned unit stride loads: 0 
   remark #15457: masked unaligned unit stride stores: 0 
   remark #15458: masked indexed (or gather) loads: 0 
   remark #15459: masked indexed (or scatter) stores: 0 
   remark #15462: unmasked indexed (or gather) loads: 0 
   remark #15463: unmasked indexed (or scatter) stores: 0 
   remark #15554: Unmasked VLS-optimized loads (each part of the group counted separately): 0 
   remark #15555: Masked VLS-optimized loads (each part of the group counted separately): 0 
   remark #15556: Unmasked VLS-optimized stores (each part of the group counted separately): 0 
   remark #15557: Masked VLS-optimized stores (each part of the group counted separately): 0 
   remark #15497: vector compress: 0 
   remark #15498: vector expand: 0 
   remark #15474: --- end vector loop memory reference summary --- 
   remark #25587: Loop has reduction 
LOOP END 
```

## Additional Information

For additional information about vectorization with the Intel oneAPI DPC++/C++ Compiler, refer to the [Vectorization](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/vectorization.html) section of the Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference.

For details about optimization report options, refer to [Optimization Report Options](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/optimization-report-options.html).

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).