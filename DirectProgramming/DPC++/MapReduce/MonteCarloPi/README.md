# `Monte Carlo Pi` Sample

Monte Carlo Simulation is a broad category of computation that utilizes statistical analysis to reach a result. This sample uses the Monte Carlo Procedure to estimate the value of pi. By inscribing a circle of radius 1 inside a 2x2 square and then sampling a large number of random coordinates falling uniformly within the square, the value of pi can be estimated using the ratio of samples that fall inside the circle divided by the total number of samples.

This method of estimation works for calculating pi because the expected value of the sample ratio is equal to the ratio of a circle's area divided by the square's: a circle of radius 1 has an area of pi units squared, while a 2x2 square has an area of 4 units squared, yielding a ratio of pi/4. Therefore, to estimate the value of pi, our solution will be four times the sample ratio.

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | How to utilize the DPC++ reduction extension
| Time to complete                  | 15 minutes


## Purpose

The Monte Carlo procedure for estimating pi is easily parallelized, as each calculation of a random coordinate point can be considered a discrete work item. The computations involved with each work item are entirely independent of one another except for in summing the total number of points inscribed within the circle. This code sample demonstrates how to utilize the DPC++ reduction extension for this purpose.

The code will attempt to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected.  The device used for the compilation is displayed in the output, along with the elapsed time to complete the computation. A rendered image plot of the computation is also written to a file.


## Key Implementation Details 

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and reduction.

## License  

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `Monte Carlo Pi` Program for CPU and GPU

> Note: if you have not already done so, set up your CLI 
> environment by sourcing  the setvars script located in 
> the root of your oneAPI installation. 
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh  
> Linux User: . ~/intel/oneapi/setvars.sh  
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat


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

2. Run the program:
    ```
    make run
    ```

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
     - Run the following command: `MSBuild MonteCarloPi.sln /t:Rebuild /p:Configuration="Release"`


## Running the Sample

### Application Parameters
constexpr int size_n =
constexpr int size_wg =

constexpr int img_dimensions =
constexpr double circle_outline =

- size_n defines the sample size for the Monte Carlo procedure. size_wg defines the size of workgroups inside the kernel code. The number of workgroups is calculated by dividing size_n by size_wg, so size_n must be greater than or equal to size_wg. Increasing size_n will increase computation time as well as the accuracy of the pi estimation. Changing size_wg will have different performance effects depending on the device used for offloading.
- img_dimensions define the size of the output image for data visualization.
- circle_outline defines the thickness of the circular border in the output image for data visualization. Setting it to zero will remove it entirely.

### Example of Output
```
Calculating estimated value of pi...

Running on Intel(R) Gen9 HD Graphics NEO
The estimated value of pi (N = 10000) is: 3.15137

Computation complete. The processing time was 0.446072 seconds.
The simulation plot graph has been written to 'MonteCarloPi.bmp'
```
