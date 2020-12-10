# STREAM Sample

This sample contains a [STREAM](http://www.cs.virginia.edu/stream/) implementation using DPC++ for CPU and GPU. 


| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04
| Hardware                          | GEN9, DG1, ATS, ICX
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | How to benchmark the memory bandwidth using STREAM.
| Time to complete                  | 5 minutes


## Purpose
The STREAM sample performs the memory bandwidth benchmark.

## Key Implementation Details 
This sample is a variance of the [STREAM](http://www.cs.virginia.edu/stream/) benchmark.
 
## License  
This code sample is licensed under MIT license. 


## Building the `STREAM` Program for CPU and GPU

### On a Linux* System 

Perform the following steps:

> Note: If you have not already done so, set up your CLI environment by sourcing 
>    the setvars script located in the root of your oneAPI installation.  
>     
>   Linux (sudo): `source /opt/intel/oneapi/setvars.sh`  
>   Linux (user): `~/intel/oneapi/ setvars.sh`  

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

## Running the Sample
./stream_sycl.exe

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
