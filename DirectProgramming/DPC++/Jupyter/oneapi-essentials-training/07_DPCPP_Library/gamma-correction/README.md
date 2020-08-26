# PSTL Gamma Correction sample

This example demonstrates PSTL with gamma correction - a nonlinear operation used to encode and decode the luminance of each image pixel. See https://en.wikipedia.org/wiki/Gamma_correction for more information.

The example creates a fractal image in memory and performs gamma correction on it. The output of the example application is a BMP image with corrected luminance.

The computations are performed using DPC++ backend of Parallel STL.


| Optimized for                   | Description                                                     |
|---------------------------------|-----------------------------------------------------------------|
| OS                              | Linux Ubuntu 18.04, Windows 10                                  |
| Hardware                        | SKL with GEN9 or newer                                          |
| Software                        | Intel&reg; oneAPI DPC++ Compiler (beta)                         |
| What you will learn             | How to offoad the computation to GPU using Intel DPC++ Compiler |
| Time to complete                | At most 5 minutes                                               |

## License

This code sample is licensed under MIT license.

## How to build

### on Linux
    mkdir build &&  
    cd build &&  
    cmake ../. &&  
    make VERBOSE=1  

### on Windows - command line
  * Build the program using MSBuild
   MSBuild gamma-correction.sln /t:Rebuild /p:Configuration="Release"

### on Windows - Visual Studio 2017 or newer
   * Open Visual Studio 2017 or newer IDE
   * Select Menu "File > Open > Project/Solution", find "gamma_correction" folder and select "gamma-correction.sln"
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program
