# `Prefix Sum` sample

This code sample demonstrates the implementation of parallel prefix sum using Intel Data Parallel C++ to
offload the computation to a GPU. In this implementation, a random sequence of 2**n elements is given
(n is a positive number) as input, the algorithm compute the prefix sum in parallel. The result sequence is
in ascending order.

For comprehensive instructions regarding DPC++ Programming, go to
https://software.intel.com/en-us/oneapi-programming-guide
and search based on relevant terms noted in the comments.
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta); Intel C++ Compiler (beta)
| What you will learn               | Implement bitonic sort using Intel DPC++ compiler
| Time to complete                  | 15 minutes


## Purpose

Given a randomized sequence of numbers x0, x1, x2, ..., xn, this algorithm computes and returns
a new sequence y0, y1, y2, ..., yn so that

y0 = x0
y1 = x0 + x1
y2 = x0 + x1 + x2
.....
yn = x0 + x1 + x2 + ... + xn

Below is the pseudo code for computing prefix sum in parallel:

n is power of 2 (1, 2, 4 , 8, 16, ...):

for i from 0 to  [log2 n] - 1 do
   for j from 0 to (n-1) do in parallel
     if j<2^i then
       x_{j}^{i+1} <- x_{j}^{i}}
     else
       x_{j}^{i+1} <- x_{j}^{i} + x_{j-2^{i}}^{i}}

In the above, the notation x_{j}^{i} means the value of the jth element of array x in timestep i.
Given n processors to perform each iteration of the inner loop in constant time, the algorithm
as a whole runs in O(log n) time, the number of iterations of the outer loop.

The code will attempt first to execute on an available GPU and fallback to the system's CPU if a
compatible GPU is not detected.

## Key Implementation Details

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command
groups.

## License  
This code sample is licensed under MIT license  

## Building the `PrefixSum` Program for CPU and GPU

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU,
FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI
Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
1. Build the program using the following `cmake` commands. 
    ``` 
    $ cd PrefixSum
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

### On a Windows* System
    * Build the program using VS2017 or VS2019
      Right click on the solution file and open using either VS2017 or VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.

    * Build the program using MSBuild
      Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for
 VS2019"
      Run - MSBuild PrefixSum.sln /t:Rebuild /p:Configuration="Release"

## Running the sample
### Application Parameters
	
        Usage: PrefixSum <exponent> <seed>

where

exponent is a positive number. The according length of the sequence is 2**exponent.

seed is the seed used by the random generator to generate the randomness.

The sample offloads the computation to GPU and then performs the verification the results in the CPU.
The results are verified if yk = yk-1 + xk the original compared. If the results are matched and
the ascending order is verified, the application will display a “Success!” message.

### Example of Output
```
$ ./PrefixSum 21 47

Sequence size: 2097152, seed: 47
Num iteration: 21
Device: Intel(R) Gen9 HD Graphics NEO
Kernel time: 170 ms

Success!
```
