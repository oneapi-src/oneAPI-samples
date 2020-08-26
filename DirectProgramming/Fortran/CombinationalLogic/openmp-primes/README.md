# `OpenMP Primes`
This sample is designed to illustrate how to use 
the OpenMP* API with the Intel® Fortran Compiler.

This program finds all primes in the first 40,000,000 integers, 
the number of 4n+1 primes, and the number of 4n-1 primes in the same range. 
It illustrates two OpenMP* directives to help speed up the code.

  
| Optimized for                     | Description
|:---                               |:---
| OS                                | macOS* with Xcode* installed 
| Software                          | Intel&reg; oneAPI Intel Fortran Compiler (Beta)
| What you will learn               | How to build and run a Fortran OpenMP application using Intel Fortran compiler
| Time to complete                  | 10 minutes

## Purpose

This program finds all primes in the first 40,000,000 integers, the number of 4n+1 primes, 
and the number of 4n-1 primes in the same range. It illustrates two OpenMP* directives 
to help speed up the code.

First, a dynamic schedule clause is used with the OpenMP* for directive. 
Because the DO loop's workload increases as its index gets bigger, 
the default static scheduling does not work well. Instead, dynamic scheduling 
is used to account for the increasing workload. 
But dynamic scheduling itself has more overhead than static scheduling, 
so a chunk size of 10 is used to reduce the overhead for dynamic scheduling.

Second, a reduction clause is used instead of an OpenMP* critical directive 
to eliminate lock overhead. A critical directive would cause excessive lock overhead 
due to the one-thread-at-time update of the shared variables each time through the DO loop. 
Instead the reduction clause causes only one update of the shared variables once at the end of the loop.

The sample can be compiled unoptimized (-O0 ), or at any level of 
optimization (-O1 through -O3 ). In addition, the following compiler options are needed.

The option -qopenmp enables compiler recognition of OpenMP* directives. 
This option can also be omitted, in which case the generated executable will be a serial program.

The option -fpp enables the Fortran preprocessor.
Read the Intel® Fortran Compiler Documentation for more information about these options.

## Key Implementation Details
The Intel&reg; oneAPI Intel Fortran Compiler (Beta) includes all libraries and headers   necessary to compile and run OpenMP* enabled Fortran applications. Users simply use the -qopenmp compiler option to compile and link their OpenMP enabled applications. 

## License  
This code sample is licensed under MIT license  

## Building the `Fortran OpenMP*` sample  

### Experiment 1: Unoptimized build and run
* Build openmp_samples

        cd openmp_samples 
        make clean 
        make debug

   * Run the program

        make debug_run

   * What did you see?

     Did the debug, unoptimized code run slower? 
     
### Experiment 2: Default Optimized build and run 

   * Build openmp_samples

    make 
   * Run the program

    make run  

### Experiment 3: Controlling number of threads
By default an OpenMP application creates and uses as many threads as there are "processors" in a system.  A "processor" is the number of logical processors which on hyperthreaded cores is twice the number of physical cores.

OpenMP uses environment variable 'OMP_NUM_THREADS' to set number of threads to use.  Try this!

    export OMP_NUM_THREADS=1
    make run
note the number of threads reported by the application.  Now try 2 threads:

    export OMP_NUM_THREADS=2
    make run
Did the make the application run faster?  Experiment with the number of threads and see how it affects performance.

### Clean up 
   * Clean the program  
    make clean

## Further Reading
Interested in learning more?  We have a wealth of information 
on using OpenMP with the Intel Fortran Compiler in our 
[OpenMP section of Developer Guide and Reference][1]

[1]: https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top/optimization-and-programming-guide/openmp-support.html "Developer Guide and Reference"
