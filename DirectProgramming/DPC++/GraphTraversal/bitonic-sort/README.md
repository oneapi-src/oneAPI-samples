# `Bitonic Sort` Sample

This code sample demonstrates the implementation of bitonic sort using Intel Data Parallel C++ to
offload the computation to a GPU. In this implementation, a random sequence of 2**n elements is given
(n is a positive number) as input, and the algorithm sorts the sequence in parallel. The result sequence is
in ascending order.

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | Implement bitonic sort using Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes


## Purpose

The algorithm converts a randomized sequence of numbers into
a bitonic sequence (two ordered sequences) and then merges these two ordered
sequences into an ordered sequence. Bitonic sort algorithm is briefly
described as followed:

- First, it decomposes the randomized sequence of size 2\*\*n into 2\*\*(n-1)
pairs where each pair consists of 2 consecutive elements. Note that each pair
is a bitonic sequence.
- Step 0: for each pair (sequence of size 2), the two elements are swapped so
that the two consecutive pairs form  a bitonic sequence in increasing order,
the next two pairs form the second bitonic sequence in decreasing order.
The next two pairs form the third bitonic sequence in increasing order, etc., ....
At the end of this step, we have 2\*\*(n-1) bitonic sequences of size 2, and
they follow an order of increasing, decreasing, increasing, .., decreasing.
Thus, they form 2\*\*(n-2) bitonic sequences of size 4.
- Step 1: for each new 2\*\*(n-2) bitonic sequences of size 4, (each new
sequence consists of 2 consecutive previous sequences), it swaps the elements
so that at the end of step 1, we have 2\*\*(n-2) bitonic sequences of size 4,
and they follow an order: increasing, decreasing, increasing, ...,
decreasing. Thus, they form 2\*\*(n-3) bitonic sequences of size 8.
- Same logic applies until we reach the last step.
- Step n: at this last step, we have one bitonic sequence of size 2\*\*n. The
elements in the sequence are swapped until we have a sequence in increasing
order.

The code will attempt first to execute on an available GPU and fallback to the system's CPU
if a compatible GPU is not detected.

## Key Implementation Details

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command g
roups. Unified Shared Memory (USM) and Buffer Object are used for data management.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `bitonic-sort` Program for CPU and GPU

> Note: if you have not already done so, set up your CLI 
> environment by sourcing  the setvars script located in 
> the root of your oneAPI installation. 
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh  
> Linux User: . ~/intel/oneapi/setvars.sh  
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU,
FPGA) as well as whether to run in batch or interactive mode. For more information, see the Intel® oneAPI
Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System
1. Build the program using the following `cmake` commands. 
    ``` 
    $ cd bitonic-sort
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
     - Run the following command: `MSBuild bitonic-sort.sln /t:Rebuild /p:Configuration="Release"'

## Running the sample
### Application Parameters
	
        Usage: bitonic-sort <exponent> <seed>

where

exponent is a positive number. The according length of the sequence is 2**exponent.

seed is the seed used by the random generator to generate the randomness.

The sample offloads the computation to GPU and then performs the computation in serial in the CPU.
The results from the parallel and serial computation are compared. If the results are matched and
the ascending order is verified, the application will display a “Success!” message.

### Example of Output
```
$ ./bitonic-sort 21 47
Array size: 2097152, seed: 47
Device: Intel(R) Gen9 HD Graphics NEO
Warm up ...
Kernel time using USM: 0.248422 sec
Kernel time using buffer allocation: 0.253364 sec
CPU serial time: 0.628803 sec

Success!
```
