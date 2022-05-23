# `Mandelbrot` Sample

Mandelbrot is an infinitely complex fractal patterning that is derived from a simple formula. This sample demonstrates how to accelerate program performance with SIMD and parallelization using OpenMP*, in the context of calculating the Mandelbrot set.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux. MacOS Catalina or newer;
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

Mandelbrot is a C++ application that generates a fractal image by tracking how many iterations of the function z_n+1 = z_n^2 + c remain bounded, where c is a coordinate on the complex plane and z_0 = 0. In performing this calculation, complex numbers belonging to the Mandelbrot set will take infinite iterations as they will always remain bounded. So a maximum depth of iterations is set so that the program may execute infinite time.

Each point on the complex plane can be calculated independently, which lends the Mandelbrot image's calculation to parallelism. Furthermore, since each point's calculation is identical, the program can take advantage of SIMD directives to get even greater performance. This code sample demonstrates how to optimize a serial implementation of the Mandelbrot image calculation using OpenMP pragmas for SIMD and parallelization.


## Key Implementation Details

The four mandelbrot function implementations are all identically written. The only difference being the progressive use of OpenMP pragmas for enabling parallelization and SIMD.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel®
oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.


## Building the `Mandelbrot` Program

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

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

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)

## Running the Sample

### Application Parameters
You can modify the Mandelbrot parameters from within the main() definition near the head. The configurable parameters allow one to modify the dimensions(resolution) of the output image, which will also affect the program's execution time. max_depth defines the upper limit of iterations the mandelbrot function will take to calculate a single point:
    int height = 1024;
    int width = 2048;
    int max_depth = 100;

In mandelbrot.cpp, the schedule(<static/dynamic>, <chunk_size>) pragmas in the OpenMP parallel for sections can be modified to change the parallelization parameters. changing between static and dynamic affects how work items are distributed between threads, and the chunk_size affects each work item's size. in mandelbrot.cpp, there is a preprocessor definition NUM_THREADS: Changing this value affects the number of threads dedicated to each parallel section. The ideal number of threads will vary based on device hardware.

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
