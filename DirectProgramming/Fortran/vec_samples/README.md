# Fortran Vectorization Sample

The Intel® Compiler has an auto-vectorizer that detects operations in the application 
that can be done in parallel and converts sequential operations 
to parallel operations by using the 
Single Instruction Multiple Data (SIMD) instruction set.

In this sample, you will use the auto-vectorizer to improve the performance 
of the sample application. You will compare the performance of the 
serial version and the version that was compiled with the auto-vectorizer. 
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | macOS* with Xcode installed (see Release Notes for details)
| Software                          | Intel&reg; oneAPI Intel Fortran Compiler (beta)
| What you will learn               | Vectorization using Intel Fortran compiler
| Time to complete                  | 15 minutes


## License  
This code sample is licensed under MIT license  

## How to Build  
   * make


### Clean up 
   * Clean the program  
    make clean

