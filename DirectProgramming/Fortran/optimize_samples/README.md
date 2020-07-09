# Fortran Optimization Sample
 
This sample is designed to illustrate specific 
compiler optimizations, features, tools, and programming concepts.

This program computes the integral (area under the curve) of a user-supplied function 
over an interval in a stepwise fashion. 
The interval is split into segments, and at each segment position the area of a rectangle 
is computed whose height is the value of sine at that point and the width is the segment width. 
The areas of the rectangles are then summed.

The process is repeated with smaller and smaller width rectangles, 
more closely approximating the true value.

The source for this program also demonstrates recommended Fortran coding practices.

## Compile the sample several times using different optimization options:

   * O1 - Enables optimizations for speed and disables some optimizations that increase code size and affect speed.
   * O2 - Enables optimizations for speed. This is the generally recommended optimization level. Vectorization is enabled at O2 and higher levels.
   * O3 - Performs O2 optimizations and enables more aggressive loop transformations such as Fusion, Block-Unroll-and-Jam, and collapsing IF statements.

Read the [Intel® Fortran Compiler Developer Guide and Reference][1]:<https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top.html>"Intel® Fortran Compiler Developer Guide and Reference" 
 for more information about these options.

Some of these automatic optimizations use features and options that can 
restrict program execution to specific architectures.  

| Optimized for                     | Description
|:---                               |:---
| OS                                | macOS* with Xcode installed (see Release Notes for details)
| Software                          | Intel&reg; oneAPI Intel Fortran Compiler (beta)
| What you will learn               | Vectorization using Intel Fortran compiler
| Time to complete                  | 15 minutes


## License  
This code sample is licensed under MIT license  

## How to Build  
Use the one of the following compiler options:


## macOS* : -O0 -O1, -O2, -O3 

### STEP 1: Build and run at O0
    cd optimize_samples 
    edit Makefile
    set optimization levels
    uncomment FC = ifort -O0 like this 
     FC = ifort -O0 
     #FC = ifort -O1 
     #FC = ifort -O2 
     #FC = ifort -O3  
    make 

   * Run the program
    make run

   * Note the final run time (example)
    CPU Time = 3.776983 seconds

   * Clean the program
    make clean

### STEP 2: Build and run at O1
    cd optimize_samples
    edit Makefile
    set optimization levels
    uncomment FC = ifort -O1 like this
     #FC = ifort -O0
     FC = ifort -O1
     #FC = ifort -O2
     #FC = ifort -O3
    make

   * Run the program
    make run

   * Note the final run time (example)
    CPU Time = 1.444569 seconds

   * Clean the program
    make clean

### STEP 3: Build and run at O2
    cd optimize_samples
    edit Makefile
    set optimization levels
    uncomment FC = ifort -O2 like this
     #FC = ifort -O0
     #FC = ifort -O1
     FC = ifort -O2
     #FC = ifort -O3
    make

   * Run the program
    make run

   * Note the final run time (example)
     CPU Time = 0.5143980 seconds

   * Clean the program
    make clean

### STEP 4: Build and run at O3
    cd optimize_samples
    edit Makefile
    set optimization levels
    uncomment FC = ifort -O3 like this
     #FC = ifort -O0
     #FC = ifort -O1
     #FC = ifort -O2
     FC = ifort -O3
    make

   * Run the program
    make run

   * Note the final run time (example)
     CPU Time = 0.5133380 seconds

   * Clean the program
    make clean

## What did we learn?
There are big jumps going from O0 to O1, and from O1 to O2. 
but very little going from O2 to O3.
This does vary by application but generally with Intel Compilers 
O2 is has most aggressive optimizations.  Sometimes O3 can help, of course,
but generally O2 is sufficient for most applications. 

### Extra Exploration
The Intel® Fortran Compiler has many options for optimization. 
If you have a genuine Intel® Architecture process, try these additional options
    edit Makefile
    set optimization levels
    uncomment FC = ifort -O3 and add additional options shown:
     #FC = ifort -O0
     #FC = ifort -O1
     #FC = ifort -O2
     FC = ifort -O3 -xhost -align array64byte
    make

   * Run the program
    make run

   * Note the final run time (example)
     CPU Time = 0.2578490 seconds

   * Clean the program
    make clean
There are 2 additional compiler options here that are worth mentioning:

Read the online [Developer Guide and Reference][1]:<https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top.html>"Developer Guide and Reference" for more information about
the options
 1. -xhost (sub option of -x option): [-x][1]:<https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/compiler-option-details/code-generation-options/x-qx.html>"-x option"
 2. -align array64byte [-align ][1]:<https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/compiler-option-details/data-options/align.html>"-align option"

### Clean up 
   * Clean the program  
    make clean

