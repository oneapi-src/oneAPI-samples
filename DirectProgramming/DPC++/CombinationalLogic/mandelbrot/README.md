# `Mandelbrot` Sample

Mandelbrot is an infinitely complex fractal patterning that is derived from a simple formula.  It demonstrates using DPC++ for offloading computations to a GPU (or other devices) and shows how processing time can be optimized and improved with parallelism.

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to offload the computation to GPU using the Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

## Purpose
Mandelbrot is a DPC++ application that generates a fractal image by initializing a matrix of 512 x 512, where the computation at each point (pixel) is entirely independent of the computation at other points. The sample includes both parallel and serial calculation of the set, allowing for a direct comparison of results. The parallel implementation can demonstrate the use of Unified Shared Memory (USM) or buffers. You can modify parameters such as the number of rows, columns, and iterations to evaluate the difference in performance and load between USM and buffers. This is further described at the end of this document in the "Running the Sample" section.

The code will attempt to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected.  The compilation device is displayed in the output along with elapsed time to render the Mandelbrot image. This helps compare different offload implementations based on the complexity of the computation. 

## Key Implementation Details 
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.
 
## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `Mandelbrot` Program for CPU and GPU

> Note: if you have not already done so, set up your CLI 
> environment by sourcing  the setvars script located in 
> the root of your oneAPI installation. 
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh  
> Linux User: . ~/intel/oneapi/setvars.sh  
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands. 
``` 
$ mkdir build
$ cd build
$ cmake ..
$ make
```

> Note: by default, executables are created for both USM and buffers. You can build individually with the following: 
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
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild mandelbrot.sln /t:Rebuild /p:Configuration="Release"`


## Running the Sample
### Application Parameters 
You can modify the Mandelbrot parameters from within mandel.hpp. The configurable parameters include:
    row_size = 
    col_size =
    max_iterations =
    repetitions =
The default row and column size are 512.  Max interatins and repetitions are both 100.  By adjusting the parameters, you can observe how the performance varies using the different offload techniques.  
> Note: If the values drop below 128 for row and column, the output is limited to just text in the output window.

### Example of Output
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
