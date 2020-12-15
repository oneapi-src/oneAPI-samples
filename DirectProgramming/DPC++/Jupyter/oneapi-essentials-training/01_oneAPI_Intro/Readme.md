The simple-vector-incr and simple-vector-add programs are simpler implementations of vector-add.  The learning objective for a developer inspecting simple-vector-incr.cpp is to follow the instructions in the code's comments. Following Steps 1 - 5, the developer will modify the code from adding element-wise +1 to an input vector and eventually adding in pieces another input vector, buffer, and accessor that adds two vectors together.  The final product should look similar to simple-vector-add.cpp.
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | The developer will learn about buffers, accessors, and command group handlers.
| Time to complete                  | 15 minutes  
  
## Key implementation details 
Implementing the simple-vector-incr program is such that a developer following Steps 1 - 5 embedded in the code will learn about the buffers, accessors, and command group handler.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## How to Build for CPU and GPU 

### on Linux*  
   * Build the program using Make  
    make all  

   * Run the program  
    make run  

   * Clean the program  
    make clean 

### On Windows*

#### Command line using MSBuild

*  MSBuild simple-vector-inc.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE

* Open Visual Studio 2017
* Select Menu "File > Open > Project/Solution", find "simple-vector-add" folder and select "simple-vector-add.sln"
* Select Menu "Project > Build" to build the selected configuration
* Select Menu "Debug > Start Without Debugging" to run the program

