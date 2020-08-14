# `Monte Carlo Pi` Sample

Monte Carlo Simulation is a broad category of computation which utlizes statistical analysis to reach a result. This sample utilizes the Monte Carlo Procedure to estimate the value of pi: By inscribing a circle of radius 1 inside a 2x2 square, and then sampling a large number of random coordinates falling uniformly within the square, we can determine the value of pi using the ratio of samples which fall inside the circle divided by the total number of samples.

This method of estimation works for calculating pi, because the ratio of samples is proportional to the ratio of a circle's area divided by the square's: a circle of radius 1 has an area of pi units squared, while a 2x2 square has an area of 4 units squared, yielding a ratio of pi/4. Therefore, to estimate the value of pi, our solution will be 4 times the sample ratio.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler beta;
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler
| Time to complete                  | 15 minutes


## Purpose

The Monte Carlo estimation is prime for acceleration using parallelization and offloading, due to the nature of its procedure; it only requires generating a large number of random coordinates and then evaluating whether or not they fall within the circle. The challenge lies in the reduction stage: because the results of each test must be summed together to get the total number of coordinates inscribed 

[_TEMPLATE: explictly note device use._]
The code will attempt first to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected.  The device used for compilation is displayed in the output along with elapsed time to render the mandelbrot image. This is helpful for comparing different offload implementations based on complexity of the computation. 


## Key Implementation Details 

[_TEMPLATE: short punch list of key terms._]
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

 
## License  

This code sample is licensed under MIT license. 


## Building the `Mandelbrot` Program for CPU and GPU

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

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
      Run - MSBuild mandelbrot.sln /t:Rebuild /p:Configuration="Release"


## Running the Sample

### Application Parameters 
[_TEMPLATE: this is an opportunity to provide pointers in the code for how the user can alter the application to adjust run times and compare performance._]
You can modify the Mandelbrot parameters from within mandel.hpp. The configurable parameters include:
    row_size = 
    col_size =
    max_iterations =
    repetitions =
The default row and column size is 512.  Max interatins and repetions are both 100.  By adjusting the parameters, you can observe how the performance varies using the different offload techniques.  Note that if the values drop below 128 for row and column, the output is limted to just text in the ouput window.

### Example of Output
[_TEMPLATE: Always provide an example of the expected outcome. This is confirmation to the user that the application worked as expected. Also, be sure your application output is complete and supportive._]
```
Platform Name: Intel(R) OpenCL HD Graphics
  Platform Version: OpenCL 2.1 
       Device Name: Intel(R) Gen9 HD Graphics NEO
    Max Work Group: 256
 Max Compute Units: 24

Parallel Mandelbrot set using buffers.
Rendered image output to file: mandelbrot.png (output too large to display in text)
       Serial time: 0.0430331s
     Parallel time: 0.00224131s
Successfully computed Mandelbrot set.
```