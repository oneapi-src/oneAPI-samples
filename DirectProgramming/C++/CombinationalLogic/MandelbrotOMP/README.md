# `Mandelbrot OMP` Sample

Mandelbrot is an infinitely complex fractal patterning that is derived from a simple formula. This sample demonstrates how to accelerate program performance with SIMD and parallelization using OpenMP* in the context of calculating the Mandelbrot set.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to optimize a scalar implementation using OpenMP pragmas
| Time to complete         | 15 minutes

## Purpose

The `Mandelbrot OMP` C++ code sample demonstrates how to optimize a serial implementation of the Mandelbrot image calculation using OpenMP pragmas for SIMD and parallelization,which generates an ASCII Portable Pixel Map (PPM) image of the Mandelbrot set, using OpenMP for parallel execution

This sample is a C++ application that generates a fractal image by tracking how many iterations of the function z(n+1) = z(n)^2 + c remain bounded (where c is a coordinate on the complex plane and $z_0 = 0$). When performing the calculation, complex numbers belonging to the Mandelbrot set will take infinite iterations as they will always remain bounded, so a maximum depth of iterations is set so that the program may execute in finite time.

Each point on the complex plane can be calculated independently, which make the Mandelbrot image calculation suitable for parallelism. Since the calculation of each point is identical, the program can take advantage of SIMD directives to get better performance.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> macOS* Catalina or newer
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

The four mandelbrot function implementations are all identically written. The only difference being the progressive use of OpenMP pragmas for enabling parallelization and SIMD.

Performance number tabulation.

| Mandelbrot Version                | Performance data
|:---                               |:---
| Scalar baseline                   | 1.0
| OpenMP SIMD                       | 2x speedup
| OpenMP parallel                   | 6x speedup
| OpenMP SIMD + parallel            | 10x speedup

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Mandelbrot OMP` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux* and macOS*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux* and macOS*
1. Change to the sample directory.
2. Build the program.
    ```
    make
    ```
    Alternatively, you can enable the performance tabulation mode by specifying `perf_num=1`.
    ```
    export perf_num=1
    make
    ```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Mandelbrot OMP` Program

### Configurable Parameters

You can modify the Mandelbrot parameters near the head of the `main.cpp` source file.

| Parameter              | Description
|:---                    |:---
|`int height = 1024;`    | Defines height of output image.
|`int width = 2048;`     | Defines width of output image.
|`int max_depth = 100;`  | Defines the upper limit of iterations the mandelbrot function will take to calculate a single point.

>**Note**: Changing the dimensions (resolution) of the output image will affect program execution time.

In `mandelbrot.cpp`, the schedule(<static/dynamic>, <chunk_size>) pragmas in the OpenMP parallel for sections can be modified to change the parallelization parameters. changing between static and dynamic affects how work items are distributed between threads, and the chunk_size affects each work item's size. In `mandelbrot.cpp`, there is a preprocessor definition `NUM_THREADS` set to **8**. Changing this value affects the number of threads dedicated to each parallel section. The ideal number of threads will vary based on device hardware.

### On Linux and macOS

1. Run the program.
   ```
   make run
   ```

2. Clean the program. (Optional)
   ```
   make clean
   ```


## Example Output
Once you run the program, you are prompted to select a test. Each test creates a corresponding image file in .png format.

Running the program in performance tabulation mode results in multiple runs of all tests and displays average execution times for serial, SIMD, parallel, and SIMD+parallel. (Results not shown below.)

The following example output shows results similar to those you get when you select `[0] all tests` for a default run.

```
This example will check how many iterations of z_n+1 = z_n^2 + c a complex
set will remain bounded. Pick which parallel method you would like to use.
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

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
