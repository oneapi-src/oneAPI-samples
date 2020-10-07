# Sparse Matrix Vector sample
`Sparse Matrix Vector sample` provides a parallel implementation of a merge based sparse matrix and vector multiplication algorithm using DPC++. 

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | The Sparse Matrix Vector sample demonstrates the following using the Intel&reg; oneAPI DPC++/C++ Compiler <ul><li>Offloading compute intensive parts of the application using lambda kernel</li><li>Measuring kernel execution time</li></ul>
| Time to complete                  | 15 minutes


## Purpose
Sparse linear algebra algorithms are common in HPC, in fields as machine learning and computational science. In this sample, a merge based sparse matrix and vector multiplication algorithm is implemented. The input matrix is in compressed sparse row format. Use a parallel merge model enables the application to efficiently offload compute intensive operation to the GPU. For comparison, the application is run sequentially and parallelly with run times for each displayed in the application's output. The device where the code is run is also identified.

The workgroup size requirement is 256.  If your hardware cannot support this, the application will present an error.

Compressed Sparse Row (CSR) representation for sparse matrix have three components:
<ul>
<li>Nonzero values</li>
<li>Column indices</li>
<li>Row offsets</li>
</ul>

Both row offsets and values indices can be thought of as sorted arrays. The progression of the computation is similar to that of merging two sorted arrays at a conceptual level.

In parallel implementation, each thread independently identifies its scope of the merge and then performs only the amount of work that belongs this thread in the cohort of threads.

## Key implementation details
Includes device selector, unified shared memory, kernel, and command groups in order to implement a solution using a parallel merge method ih which each thread independently identifies its scope of the merge and then performs only the amount of work that belongs this thread. 


## License  
This code sample is licensed under MIT license. 

## Building the Program for CPU and GPU

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.


### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel&reg; oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System

Perform the following steps:

1.  Build the program using the following `cmake` commands.
```
    $ cd merge-spmv
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

2.  Run the program 
```
    $ make run

```

### On a Windows* System Using Visual Studio* version 2017 or Newer

* Build the program using VS2017 or VS2019: Right click on the solution file and open using either VS2017 or VS2019 IDE. Right click on the project in Solution explorer and select Rebuild. From top menu select Debug -> Start without Debugging.
* Build the program using MSBuild: Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019". Run - MSBuild merge-spmv.sln /t:Rebuild /p:Configuration="Release"


## Running the sample

### Example Output
```
Device: Intel(R) Gen9
Compute units: 24
Work group size: 256
Repeating 16 times to measure run time ...
Iteration: 1
Iteration: 2
Iteration: 3
...
Iteration: 16
Successfully completed sparse matrix and vector multiplication!
Time sequential: 0.00436269 sec
Time parallel: 0.00909913 sec

```
