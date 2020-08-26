# `Mandelbrot` Sample

Mandelbrot is an infinitely complex fractal patterning that is derived from a simple formula. This sample demonstrates how to accelerate program performance with SIMD and parallelization using OpenMP*, in the context of calculating the mandelbrot set.


| Optimized for                     | Description
|:---                               |:---
| OS                                | MacOS Catalina or newer;
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI C++ Compiler Classic
| What you will learn               | How to optimize a scalar implementation using OpenMP pragmas
| Time to complete                  | 15 minutes

Performance number tabulation

| Mandelbrot Version                | Performance data
|:---                               |:---
| Scalar baseline                   | 1.0
| OpenMP SIMD                       | 2x speedup
| OpenMP parallel                   | 6x speedup
| OpenMP SIMD + parallel            | 10x speedup


## Purpose

Mandelbrot is a C++ application that generates a fractal image by tracking how many iterations of the function z_n+1 = z_n^2 + c remain bounded, where c is a coordinate on the complex plane and z_0 = 0. In performing this calculation, complex numbers belonging to the mandelbrot set will take infinite iterations as they will always remain bounded. So a maximum depth of iterations is set so that the program may execute in finite time.

Each point on the complex plane can be calculated independently, which lends the calculation of the mandelbrot image to parallelism. Furthermore, since the calculation on each point is identical, the program can take advantage of SIMD directives to get even greater performance. This code sample demonstrates how to optimize a serial implementation of the mandelbrot image calculation using OpenMP pragmas for SIMD and parallelization.


## Key Implementation Details 

The 4 mandelbrot function implementations are identically written, with the only difference being the progressive use of OpenMP pragmas for enabling parallelization and SIMD.

 
## License  

This code sample is licensed under MIT license. 


## Building the `Mandelbrot` Program

Perform the following steps:
1. Build the program using the following `make` commands. 
``` 
$ export perf_num=1     *optional, will enable performance tabulation mode
$ make
```

2. Run the program:
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```


## Running the Sample

### Application Parameters 
You can modify the Mandelbrot parameters from within the main() definition near the head. The configurable parameters allow one to modify the dimensions(resolution) of the output image, which will also affect the execution time of the program. max_depth defines the upper limit of iterations the mandelbrot function will take to calculate a single point:
    int height = 1024;
    int width = 2048;
    int max_depth = 100;

In mandelbrot.cpp, the schedule(<static/dynamic>, <chunk_size>) pragmas in the OpenMP parallel for sections can me modified to change the parallelization parameters. changing between static and dynamic affects how work items are distributed between threads, and the chunk_size affects the size of each work item. On line 69, there is a preprocessor definition NUM_THREADS: Changing this value affects the number of threads dedicated to each parallel section. The ideal number of threads will vary based on the device hardware.

### Example of Output
```
This example will check how many iterations of z_n+1 = z_n^2 + c a complex set will remain bounded. Pick which parallel method you would like to use.
[0] all tests
[1] serial/scalar
[2] OpenMP SIMD
[3] OpenMP Parallel
[4] OpenMP Both
  > 0

Running all tests

Starting serial, scalar Mandelbrot...
Calculation finished. Processing time was 198ms
Saving image as mandelbrot_serial.png

Starting OMP SIMD Mandelbrot...
Calculation finished. Processing time was 186ms
Saving image as mandelbrot_simd.png

Starting OMP Parallel Mandelbrot...
Calculation finished. Processing time was 33ms
Saving image as mandelbrot_parallel.png

Starting OMP SIMD + Parallel Mandelbrot...
Calculation finished. Processing time was 31ms
Saving image as mandelbrot_simd_parallel.png
```