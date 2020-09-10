# `Matrix Multiplication` sample
Matrix Multiplication is a simple program that multiplies together two large matrices and verifies the results.
This program is implemented using C++ with oneAPI Math Kernel Library (oneMKL):

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10*
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler beta, oneMKL
| What you will learn               | Offloads computations on 2D arrays to GPU using oneMKL
| Time to complete                  | 15 minutes

## Key implementation details
oneMKL implementation explained.

## License
This code sample is licensed under MIT license.

## Bulding `Matrix Multiplication` for oneMKL

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands. 
``` 
$ mkdir build
$ cd build
$ cmake ..
$ make
```

> Note: by default, exectables are created for both USM and buffers. You can build individually with the following: 
>    Create buffers executable: make mandelbrot
>    Create USM executable: make mandelbrot_usm

2. Run the program (default uses buffers):
    ```
    make run
    ```
> Note: for USM use `make run_usm`

3. Clean the program using:
    ```
    make clean
    ```

### On a Windows* System Using Visual Studio* Version 2017 or Newer

* Build the program using VS2017 or VS2019
      Right click on the solution file and open using either VS2017 or VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.


* Build the program using MSBuild
      Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
      Run - MSBuild matrix_mul.sln /t:Rebuild /p:Configuration="Release"

#### Visual Studio IDE
   * Open Visual Studio 2017
   * Select Menu "File > Open > Project/Solution", find "matrix_mul" folder and select "matrix_mul.sln"
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program

