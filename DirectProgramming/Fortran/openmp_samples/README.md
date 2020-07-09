# Fortran OpenMP sample
This sample is designed to illustrate how to use 
the OpenMP* API with the Intel® Fortran Compiler.

This program finds all primes in the first 10,000,000 integers, 
the number of 4n+1 primes, and the number of 4n-1 primes in the same range. 
It illustrates two OpenMP* directives to help speed up the code.


This program finds all primes in the first 10,000,000 integers, the number of 4n+1 primes, 
and the number of 4n-1 primes in the same range. It illustrates two OpenMP* directives 
to help speed up the code.

First, a dynamic schedule clause is used with the OpenMP* for directive. 
Because the for loop's workload increases as its index gets bigger, 
the default static scheduling does not work well. Instead, dynamic scheduling 
is used to account for the increasing workload. 
But dynamic scheduling itself has more overhead than static scheduling, 
so a chunk size of 10 is used to reduce the overhead for dynamic scheduling.

Second, a reduction clause is used instead of an OpenMP* critical directive 
to eliminate lock overhead. A critical directive would cause excessive lock overhead 
due to the one-thread-at-time update of the shared variables each time through the for loop. 
Instead the reduction clause causes only one update of the shared variables once at the end of the loop.

The sample can be compiled unoptimized (-O0 ), or at any level of 
optimization (-O1 through -O3 ). In addition, the following compiler options are needed.

The option -qopenmp enables compiler recognition of OpenMP* directives. 
This option can also be omitted, in which case the generated executable will be a serial program.

The option -fpp enables the Fortran preprocessor.
Read the Intel® Fortran Compiler Documentation for more information about these options.
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | macOS* with Xcode installed (see Release Notes for details)
| Software                          | Intel&reg; oneAPI Intel Fortran Compiler (beta)
| What you will learn               | How to build and run a Fortran OpenMP application using Intel Fortran compiler
| Time to complete                  | 10 minutes


## License  
This code sample is licensed under MIT license  

## How to Build  

### Experiment 1 Default Optimized build and run 
   * Build openmp_samples 

    cd openmp_samples &&
    make clean &&
    make

   * Run the program

    make run  

### Experiment 2 Unoptimized build and run
   * Build openmp_samples

    cd openmp_samples &&
    make clean &&
    make debug

   * Run the program

    make debug_run

   * What did you see?

     Did the debug, unoptimized code run slower? 


### Clean up 
   * Clean the program  
    make clean

