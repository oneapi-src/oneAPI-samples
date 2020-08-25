The simple-vector-incr and simple-vector-add programs are simpler implementations of vector-add.  The learning objective for a developer inspecting simple-vector-incr.cpp is to follow the instructions in the comments within the code. Following Steps 1 - 5 the developer will modify the code from that of adding element-wise +1 to an input vector and eventually adding in pieces another input vector, buffer, and accessor that adds two vectors together.  The final product should look similar to simple-vector-add.cpp.
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | The developer will learn about buffers, accessors, and command group handlers.
| Time to complete                  | 15 minutes  
  
## Key implementation details 
The implementation of the simple-vector-incr program is such that a developer following Steps 1 - 5 embedded in the code will learn abou the buffers, accessors, and command group handler.

## License  
This code sample is licensed under MIT license. 

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

#### Notices and Disclaimers

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document.

This document contains information on products, services and/or processes in development. All information provided here is subject to change without notice. Contact your Intel representative to obtain the latest forecast, schedule, specifications and roadmaps.

Intel technologies' features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. No product or component can be absolutely secure. Check with your system manufacturer or retailer or learn more at [intel.com]. 

The products and services described may contain defects or errors which may cause deviations from published specifications. Current characterized errata are available on request.

Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.

Intel, the Intel logo and Xeon are trademarks of Intel Corporation in the U.S. and/or other countries.

Microsoft, Windows, and the Windows logo are trademarks, or registered trademarks of Microsoft Corporation in the United States and/or other countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission of The Khronos Group.

*Other names and brands may be claimed as the property of others.

© Intel Corporation.
