# `Vectorize VecMatMult`

In this sample, you will use the auto-vectorizer to improve the performance 
of the sample application. You will compare the performance of the 
serial version and the version that was compiled with the auto-vectorizer. 
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | macOS* with Xcode* installed 
| Hardware							   | Intel-based Mac*
| Software                          | Intel&reg; oneAPI Intel Fortran Compiler (beta)
| What you will learn               | Vectorization using Intel Fortran compiler
| Time to complete                  | 15 minutes


## Purpose
The Intel® Compiler has an auto-vectorizer that detects operations in the application 
that can be done in parallel and converts sequential operations 
to parallel operations by using the 
Single Instruction Multiple Data (SIMD) instruction set.

For the Intel® compiler, vectorization is the unrolling of a loop combined with the generation of packed SIMD instructions. Because the packed instructions operate on more than one data element at a time, the loop can execute more efficiently. It is sometimes referred to as auto-vectorization to emphasize that the compiler automatically identifies and optimizes suitable loops on its own.

Intel® Advisor can assist with vectorization and show optimization report messages with your source code. See [Intel Advisor][1] for details.
[1]: https://software.intel.com/content/www/us/en/develop/tools/advisor.html "Intel Avisor"

Vectorization may call library routines that can result in additional performance gain on Intel microprocessors than on non-Intel microprocessors. The vectorization can also be affected by certain options, such as m or x.

Vectorization is enabled with the compiler at optimization levels of O2 (default level) and higher for both Intel® microprocessors and non-Intel® microprocessors. Many loops are vectorized automatically, but in cases where this doesn't happen, you may be able to vectorize loops by making simple code modifications. In this sample, you will:

1. establish a performance baseline

2. generate a vectorization report

3. improve performance by aligning data

4. improve performance using Interprocedural Optimization

## Key Implementation Details

In this sample, you will use the following files:

    driver.f90

    matvec.f90


## License  
This code sample is licensed under MIT license  


## Building the `Fortran Vectorization` sample

This sample contains 2 Fortran source files, in subdirectory 'src/' under the main sample root directory oneAPI-samples/DirectProgramming/Fortran/vectorize-vecmatmult

1. matvec.f90 is a Fortran source file with a matrix-times-vector algorithm
2. driver.f90 is a Fortran source file with the main program calling matvec

## Running the `Fortran Vectorization` sample

### Step1 Establishing a Performance Baseline

To set a performance baseline for the improvements that follow in this sample, compile your sources from the src directory with these compiler options:

    ifort -real-size 64 -O1 matvec.f90 driver.f90 -o MatVector

Execute 'MatVector'

     ./MatVector
and record the execution time reported in the output. This is the baseline against which subsequent improvements will be measured.


### Step 2 Generating a Vectorization Report

A vectorization report shows what loops in your code were vectorized and explains why other loops were not vectorized. To generate a vectorization report, use the **qopt-report-phase=vec** compiler options together with **qopt-report=1** or **qopt-report=2**.

Together with **qopt-report-phase=vec**, **qopt-report=1** generates a report with the loops in your code that were vectorized while **qopt-report-phase=vec** with **qopt-report=2** generates a report with both the loops in your code that were vectorized and the reason that other loops were not vectorized.

Because vectorization is turned off with the **O1** option, the compiler does not generate a vectorization report. To generate a vectorization report, compile your project with the **O2**, **qopt-report-phase=vec**, **qopt-report=1** options:

    ifort -real-size 64 -O2 -qopt-report=1 -qopt-report-phase=vec matvec.f90 driver.f90 -o MatVector

Recompile the program and then execute MatVector. Record the new execution time. The reduction in time is mostly due to auto-vectorization of the inner loop at line 32 noted in the vectorization report **matvec.optrpt** :

    Begin optimization report for: matvec_

      Report from: Vector optimizations [vec]


    LOOP BEGIN at matvec.f90(26,3)
      remark #25460: No loop optimizations reported

      LOOP BEGIN at matvec.f90(26,3)
        remark #15300: LOOP WAS VECTORIZED
      LOOP END

      LOOP BEGIN at matvec.f90(26,3)
      <Remainder loop for vectorization>
      LOOP END
    LOOP END

    LOOP BEGIN at matvec.f90(27,3)
      remark #25460: No loop optimizations reported

      LOOP BEGIN at matvec.f90(32,6)
      <Peeled loop for vectorization>
      LOOP END

      LOOP BEGIN at matvec.f90(32,6)
        remark #15300: LOOP WAS VECTORIZED
      LOOP END

      LOOP BEGIN at matvec.f90(32,6)
      <Alternate Alignment Vectorized Loop>
      LOOP END

      LOOP BEGIN at matvec.f90(32,6)
      <Remainder loop for vectorization>
      LOOP END
    LOOP END

Note

Your line and column numbers may be different.

**qopt-report=2** with **qopt-report-phase=vec,loop** returns a list that also includes loops that were not vectorized or multi-versioned, along with the reason that the compiler did not vectorize them or multi-version the loop.

Recompile your project with the **qopt-report=2** and **qopt-report-phase=vec,loop** options.

    ifort -real-size 64 -O2 -qopt-report-phase=vec -qopt-report=2 matvec.f90 driver.f90 -o MatVector

The vectorization report matvec.optrpt indicates that the loop at line 33 in matvec.f90 did not vectorize because it is not the innermost loop of the loop nest.

    LOOP BEGIN at matvec.f90(27,3)
      remark #15542: loop was not vectorized: inner loop was already vectorized

      LOOP BEGIN at matvec.f90(32,6)
       <Peeled loop for vectorization>
      LOOP END

      LOOP BEGIN at matvec.f90(32,6)
        remark #15300: LOOP WAS VECTORIZED
      LOOP END

      LOOP BEGIN at matvec.f90(32,6)
       <Alternate Alignment Vectorized Loop>
      LOOP END

      LOOP BEGIN at matvec.f90(32,6)
       <Remainder loop for vectorization>
         remark #15335: remainder loop was not vectorized: vectorization possible but seems inefficient. Use vector always directive or -vec-threshold0 to override 
      LOOP END
    LOOP END

Note: Your line and column numbers may be different.

For more information on the **qopt-report** and **qopt-report-phase** compiler options, see the 
[Compiler Options section][3] in the Intel® Fortran Compiler Developer Guide and Reference.
[3]: https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/alphabetical-list-of-compiler-options.html "Options"


### Step 3 Improving Performance by Aligning Data

The vectorizer can generate faster code when operating on aligned data. In this activity you will improve the vectorizer performance by aligning the arrays a, b, and c in **driver.f90** on a 16-byte boundary so the vectorizer can use aligned load instructions for all arrays rather than the slower unaligned load instructions and can avoid runtime tests of alignment. Using the ALIGNED macro will insert an alignment directive for a, b, and c in driver.f90 with the following syntax:

    !dir$ attributes align : 16 :: a,b,c

This instructs the compiler to create arrays that it are aligned on a 16-byte boundary, which should facilitate the use of SSE aligned load instructions.

In addition, the column height of the matrix a needs to be padded out to be a multiple of 16 bytes, so that each individual column of a maintains the same 16-byte alignment. In practice, maintaining a constant alignment between columns is much more important than aligning the start of the arrays.

To derive the maximum benefit from this alignment, we also need to tell the vectorizer it can safely assume that the arrays in matvec.f90 are aligned by using the directive

    !dir$ vector aligned
    
Note If you use **!dir$ vector aligned**, you must be sure that all the arrays or subarrays in the loop are 16-byte aligned. Otherwise, you may get a runtime error. Aligning data may still give a performance benefit even if **!dir$ vector aligned** is not used. See the code under the ALIGNED macro in **matvec.f90**

If your compilation targets the Intel® AVX-512 instruction set, you should try to align data on a 64-byte boundary. This may result in improved performance. In this case, **!dir$ vector aligned** advises the compiler that the data is 64-byte aligned.

Recompile the program after adding the ALIGNED macro to ensure consistently aligned data:

    ifort -real-size 64 -qopt-report=2 -qopt-report-phase=vec -D ALIGNED matvec.f90 driver.f90 -o MatVector


### Step 4 Improving Performance with Interprocedural Optimization

The compiler may be able to perform additional optimizations if it is able to optimize across source line boundaries. These may include, but are not limited to, function inlining. This is enabled with the **-ipo** option.

Recompile the program using the **-ipo** option to enable interprocedural optimization.

    ifort -real-size 64 -qopt-report=2 -qopt-report-phase=vec -D ALIGNED -ipo matvec.f90 driver.f90 -o MatVector

Note that the vectorization messages now appear at the point of inlining in **driver.f90** (line 70) and this is found in the file **ipo_out.optrpt**.

    LOOP BEGIN at driver.f90(73,16)
       remark #15541: loop was not vectorized: inner loop was already vectorized

       LOOP BEGIN at matvec.f90(32,3) inlined into driver.f90(70,14)
          remark #15398: loop was not vectorized: loop was transformed to memset or memcpy
       LOOP END

       LOOP BEGIN at matvec.f90(33,3) inlined into driver.f90(70,14)
          remark #15541: loop was not vectorized: inner loop was already vectorized

          LOOP BEGIN at matvec.f90(38,6) inlined into driver.f90(70,14)
             remark #15399: vectorization support: unroll factor set to 4
             remark #15300: LOOP WAS VECTORIZED
          LOOP END
       LOOP END
    LOOP END 


Note: Your line and column numbers may be different.

Now, run the executable and record the execution time. 

### Additional Exercises

The previous examples made use of double precision arrays. They may be built instead with single precision arrays by changing the command-line option **-real-size 64** to **-real-size 32**. The non-vectorized versions of the loop execute only slightly faster the double precision version; however, the vectorized versions are substantially faster. This is because a packed SIMD instruction operating on a 32-byte vector register operates on eight single precision data elements at once instead of four double precision data elements.

Note: In the example with data alignment, you will need to set ROWBUF=3 to ensure 16-byte alignment for each row of the matrix a. Otherwise, the directive **!dir$ vector aligned** will cause the program to fail.

This completes the sample that shows how the compiler can optimize performance with various vectorization techniques.

