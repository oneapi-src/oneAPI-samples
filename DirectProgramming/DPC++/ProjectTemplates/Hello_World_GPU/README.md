# "Hello World GPU" sample

This project is a template designed to help you create your own Data Parallel
C++ application for GPU targets. The template assumes the use of make to build
your application. Review the main.cpp source file for help with the header
files you should include and how to implement "device selector" code for
targeting your application's runtime device

If GPU is not available on your system, you can fallback to cpu or default
device.
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Windows 10
| Hardware                          | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software                          | Intel(R) oneAPI DPC++ Compiler (beta) 
| What you will learn               | Get started with DPC++ for GPU projects
| Time to complete                  | n/a

## Key implementation details 
DPC++ implementation explained. 

## License  
This code sample is licensed under MIT license.

## How to Build  

### on Windows
    * Build the program using VS2017 or VS2019
      Right click on the solution file and open using either VS2017 or 
      VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.

    * Build the program using MSBuild
      Open "x64 Native Tools Command Prompt for VS2017" or 
      "x64 Native Tools Command Prompt for VS2019"
      Run - MSBuild Hello_World_GPU.sln /t:Rebuild /p:Configuration="Release"
