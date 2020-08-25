# `Intrinsics` Sample

The intrinsic samples are designed to show how to utilize the intrinsics supported by the Intel&reg; C++ compiler in a variety of applications. The src folder contains three .cpp source files each demonstrating different functionality of the intrinsics, including vector operations, complex numbers computations, and FTZ/DAZ flags.

| Optimized for                     | Description
|:---                               |:---
| OS                                | MacOS* Catalina* or newer
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI C++ Compiler Classic
| What you will learn               | How to utlize intrinsics supported by the Intel&reg; oneAPI C++ Compiler Classic
| Time to complete                  | 15 minutes


## Purpose

Intrinsics are assembly-coded functions that allow you to use C++ function calls and variables in place of assembly instructions. Intrinsics are expanded inline, eliminating function call overhead. While providing the same benefits as using inline assembly, intrinsics improve code readability, assist instruction scheduling, and help when debugging. They provide access to instructions that cannot be generated using the standard constructs of the C and C++ languages, and allow code to leverage performance enhancing features unique to specific processors.

Further information on intriniscs can be found here: https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics.html#intrinsics_GUID-D70F9A9A-BAE1-4242-963E-C3A12DE296A1

## Key Implementation Details 

This sample makes use of intrinsic functions to perform common mathematical operations including:
- Computing a dot product of two vectors
- Computing the product of two complex numbers
The implementations include multiple functions to accomplish these tasks, each one leveraging a different set of intrinsics available to Intel&reg; processors.

 
## License  

This code sample is licensed under MIT license. 


## Building the `Intrinsics` Program

Perform the following steps:
1. Build the program using the following `make` commands. 
``` 
$ make (or "make debug" to compile with the -g flag)
```

2. Run the program:
    ```
    make run (or "make debug_run" to run the debug version)
    ```

3. Clean the program using:
    ```
    make clean
    ```


### Application Parameters 

These intrinsics samples have relatively few modifiable parameters. However, certain options are avaiable to the user:

1. intrin_dot_sample: Line 35 defines the size of the vectors used in the dot product computation.

2. intrin_double_sample: Lines 244-247 define the values of the two complex numbers used in the computation.

3. intrin_ftz_sample: This sample has no modifiable parameters.

### Example of Output
```
Dot Product computed by C:  4324.000000
Dot Product computed by C + SIMD:  4324.000000
Dot Product computed by Intel(R) SSE3 intrinsics:  4324.000000
Dot Product computed by Intel(R) AVX2 intrinsics:  4324.000000
Dot Product computed by Intel(R) AVX intrinsics:  4324.000000
Dot Product computed by Intel(R) MMX(TM) intrinsics:  4324
Complex Product(C):             23.00+ -2.00i
Complex Product(Intel(R) AVX2): 23.00+ -2.00i
Complex Product(Intel(R) AVX):  23.00+ -2.00i
Complex Product(Intel(R) SSE3): 23.00+ -2.00i
Complex Product(Intel(R) SSE2): 23.00+ -2.00i
FTZ is set.
DAZ is set.
```