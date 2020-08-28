# Make based FPGA Project Template

This project is a template designed to help you quickly create your own Data Parallel
C++ application for FPGA targets. The template assumes the use of make to
build your application. See the supplied `Makefile` file for hints
regarding the compiler options and libraries needed to compile a Data Parallel
C++ application for FPGA targets. Review the `main.cpp` source file for
help with the header files you should include and how to implement
"device selector" code for targeting your application's runtime device. 

To see a simple FPGA kernel and an explanation of the recommended workflow with FPGA 
targets, consult the "compile flow" FPGA Tutorial. 

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu LTS 18.04
| Hardware                          | Intel Programmable Acceleration Card with Intel Arria10 GX FPGA or Stratix(R) 10 SX FPGA
| Software                          | Intel(R) oneAPI DPC++ Compiler (beta), Intel(R) FPGA Add-on for oneAPI Base Toolkit
| What you will learn               | Get started with basic setup for FPGA projects
| Time to complete                  | n/a

## License

This code sample is licensed under the MIT license

## How to Build on Linux

The following instructions assume you are in the project's root folder. 
Since the template does not define a kernel, all code in "main" is executed on the host regardless of the make target.

To build the template for FPGA Emulator (fast compile time, targets a CPU-emulated FPGA device):
  ```
  make build_emu
  ```

To generate an FPGA optimization report (fast compile time, partial FPGA HW compile):
  ```
  make report
  ```
Locate the FPGA optimization report, `report.html`, in the `fpga_report.prj/reports/` directory. 

To build the template for FPGA Hardware (takes about one hour, system must
have at least 32 GB of physical dynamic memory):
  ```
  make build_hw
  ```

To run the template on FPGA Emulator:
  ```
  make run_emu
  ```

To run the template on FPGA Hardware:
  ```
  make run_hw
  ```

To clean the build artifacts use:
  ```
  make clean
  ```

## Building the Tutorial in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*).
