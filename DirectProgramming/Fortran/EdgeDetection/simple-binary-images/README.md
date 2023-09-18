# Simple Edge Detection Sample
Segmentation is a common operation in image processing to find the boundaries of objects in an image.
This sample implements a simple edge detection algorithm to find object boundaries in a binary image.
However, this sample is more about offloading Fortran code to a GPU than it is about edge detection.
The algorithm is implemented in two different but functionally equivalent ways. First, it is implemented
using ordinary nested for-loops that are parallelized using OpenMP directives. Second, it is implemented
using a single DO CONCURRENT loop, which is parallelized using the OpenMP backend. In either case, the
Intel&reg; OpenMP runtime library is capable of offloading the edge detection loops to a GPU.

| Optimized for       | Description
|:---                 |:---
| OS                  | Linux* Ubuntu* 18.04 or newer
| Hardware            | Intel&reg; CPUs and GPUs
| Software            | Intel&reg; Fortran Compiler
| What you will learn | How to offload Fortran loops to a GPU
| Time to complete    | 15 minutes

## Purpose
This sample demonstrates two Fortran implementations of edge detection:

 1. img_seg_omp_target.F90 implements edge detection on binary images using ordinary for-loops and OpenMP target directives
 2. img_seg_do_concurrent.F90 implements edge detection on binary images using only a DO CONCURRENT loop

The implementations are functionally equivalent. In both cases, the OpenMP runtime library is used to parallelize the
edge detection loops, regardless of whether they are run on the CPU or offloaded to a GPU.

## Key Implementation Details
[Using Fortran DO CONCURRENT for Accelerator Offload](https://www.intel.com/content/www/us/en/developer/articles/technical/using-fortran-do-current-for-accelerator-offload.html) provides more detailed descriptions of each example code, and discusses the relative merits of each approach.

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Building and Running this sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On a Linux System
Run `make` to build and run the sample. Six programs are generated:

 1. img_seg_cpu runs the for-loop implementation sequentially on the CPU
 2. img_seg_omp_cpu runs the for-loops in parallel on the CPU using OpenMP directives
 3. img_seg_omp_gpu offloads the for-loop in parallel on the GPU using OpenMP target directives
 4. img_seg_do_conc_cpu_seq runs the DO CONCURRENT implementation sequentially on the CPU
 5. img_seg_do_conc_cpu_par runs the DO CONCURRENT loop in parallel on the CPU
 6. img_seg_do_conc_gpu offloads the DO CONCURRENT loop to the GPU using the OpenMP backend

You can remove all generated files with `make clean`.

### Example of Output
If everything is working correctly, each example program will perform edge detection on a small, randomly-generated binary
image. It will display the original image followed by the outline of the objects in the image, e.g.:
```
OMP_TARGET_OFFLOAD=MANDATORY ./img_seg_omp_gpu -n 12 -o 2 -i 1 -d
 Grid dimensions:          12
 Number of images to process:           1
 Number of objects in each image:           2

 Binary image:
  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0
  0  1  1  1  1  1  0  0  0  0  0  0
  0  1  1  1  1  1  0  0  0  0  0  0
  0  1  1  1  1  1  0  0  0  0  0  0
  0  1  1  1  1  1  0  0  0  0  0  0
  0  1  1  1  1  1  0  0  0  0  0  0
  0  0  0  0  0  0  1  1  1  0  0  0
  0  0  0  0  0  0  1  1  1  0  0  0
  0  0  0  0  0  0  1  1  1  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0

 Edge mask:
  -  -  -  -  -  -  -  -  -  -  -  -
  -  -  -  -  -  -  -  -  -  -  -  -
  -  -  -  -  -  -  -  -  -  -  -  -
  -  T  T  T  T  T  -  -  -  -  -  -
  -  T  -  -  -  T  -  -  -  -  -  -
  -  T  -  -  -  T  -  -  -  -  -  -
  -  T  -  -  -  T  -  -  -  -  -  -
  -  T  T  T  T  T  -  -  -  -  -  -
  -  -  -  -  -  -  T  T  T  -  -  -
  -  -  -  -  -  -  T  -  T  -  -  -
  -  -  -  -  -  -  T  T  T  -  -  -
  -  -  -  -  -  -  -  -  -  -  -  -
 Image           1 took  9.010000000000000E-004 seconds
 Total time (not including first iteration):  0.000000000000000E+000 seconds
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
