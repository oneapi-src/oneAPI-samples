# STREAM Sample

This package contains a modified version of the [Stream Benchmark](http://www.cs.virginia.edu/stream/) implementation using DPC++ for CPU and GPU.


| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04
| Hardware                          | GEN9, Iris-Xe Max
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | How to benchmark the memory bandwidth using STREAM.
| Time to complete                  | 5 minutes


## Purpose
The STREAM sample performs the memory bandwidth benchmark.

## Key Implementation Details
This sample contains a STREAM implementation using DPC++ for CPU and GPU and is a variant of the [STREAM](http://www.cs.virginia.edu/stream/) benchmark code. Please review the license terms regarding publishing benchmarks.”

## License
Please note: **_“This package contains a modified version of the Stream Benchmark.”_**

For the original [Stream License]( http://www.cs.virginia.edu/stream/FTP/Code/LICENSE.txt.), which is copied below for reference

***
 Copyright 1991-2003: John D. McCalpin

 License:
  1. You are free to use this program and/or to redistribute
     this program.
  2. You are free to modify this program for your own use,
     including commercial use, subject to the publication
     restrictions in item 3.
  3. You are free to publish results obtained from running this
     program, or from works that you derive from this program,
     with the following limitations:

     3a. In order to be referred to as "STREAM benchmark results",
         published results must be in conformance to the STREAM
         Run Rules, (briefly reviewed below) published at
         http://www.cs.virginia.edu/stream/ref.html
         and incorporated herein by reference.
         As the copyright holder, John McCalpin retains the
         right to determine conformity with the Run Rules.

     3b. Results based on modified source code or on runs not in
         accordance with the STREAM Run Rules must be clearly
         labelled whenever they are published.  Examples of
         proper labelling include:
         "tuned STREAM benchmark results"
         "based on a variant of the STREAM benchmark code"
         Other comparable, clear and reasonable labelling is
         acceptable.

     3c. Submission of results to the STREAM benchmark web site
         is encouraged, but not required.
  4. Use of this program or creation of derived works based on this
     program constitutes acceptance of these licensing restrictions.
  5. Absolutely no warranty is expressed or implied.
***

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to
this readme for instructions on how to build and run a sample.

## Building the `STREAM` Program for CPU and GPU

### On a Linux* System
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

1. Build the program using the following `cmake` commands.
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
2. Run the program (default uses buffers):
    ```
    make run
    ```
3. Clean the program using:
    ```
    make clean
    ```


If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Running the Sample
```
./stream_sycl.exe
```

### Example of Output
```
$ ./stream_sycl.exe
SYCL Platform: Intel(R) Level-Zero
SYCL Device:   Intel(R) Graphics Gen9 [0x3ea5]
-------------------------------------------------------------
STREAM version $Revision: 5.10 $
-------------------------------------------------------------
This system uses 8 bytes per array element.
-------------------------------------------------------------
Array size = 134217728 (elements), Offset = 0 (elements)
Memory per array = 1024.0 MiB (= 1.0 GiB).
Total memory required = 3072.0 MiB (= 3.0 GiB).
Each kernel will be executed 20 times.
 The *best* time for each kernel (excluding the first iteration)
 will be used to compute the reported bandwidth.
-------------------------------------------------------------
Your clock granularity/precision appears to be 1 microseconds.
Each test below will take on the order of 95882 microseconds.
   (= 95882 clock ticks)
Increase the size of the arrays if this shows that
you are not getting at least 20 clock ticks per test.
-------------------------------------------------------------
WARNING -- The above is only a rough guideline.
For best results, please be sure you know the
precision of your system timer.
-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:           29330.4     0.073382     0.073217     0.074268
Scale:          28580.6     0.075360     0.075138     0.076204
Add:            27674.2     0.116595     0.116398     0.116940
Triad:          27324.7     0.118042     0.117887     0.118365
-------------------------------------------------------------
Solution Validates: avg error less than 1.000000e-13 on all three arrays
-------------------------------------------------------------

```
