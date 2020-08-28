# CMake based GPU Project Template

This project is a template designed to help you create your own Data Parallel
C++ application for GPU targets. The template assumes the use of CMake to
build your application. See the supplied `CMakeLists.txt` file for hints
regarding the compiler options and libraries needed to compile a Data Parallel
C++ application for GPU targets. And review the `main.cpp` source file for
help with the header files you should include and how to implement
"device selector" code for targeting your application's runtime device.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu LTS 18.04, 19.10; RHEL 8.x
| Hardware                          | Integrated Graphics from Intel (GPU) GEN9 or higher
| Software                          | Intel(R) oneAPI DPC++ Compiler (beta)
| What you will learn               | Get started with compile flow for GPU projects
| Time to complete                  | n/a

## License

This code sample is licensed under MIT license

## How to Build  on Linux

The following instructions assume you are in the root of the project folder.

```
    mkdir build
    cd build
    cmake ..
```
  To build the template using:
```
    make all or make build
```

  To run the template using:
```
    make run
```

## Building the Tutorial in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*).