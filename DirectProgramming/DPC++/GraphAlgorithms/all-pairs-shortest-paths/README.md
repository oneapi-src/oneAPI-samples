# All Pairs Shortest Paths sample
`All Pairs Shortest Paths` uses the Floyd-Warshall algorithm to find the shortest paths between pairs of vertices in a graph. It uses a parallel blocked algorithm that enables the application to efficiently offload compute intensive work to the GPU.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | The All Pairs Shortest Paths sample demonstrates the following using the Intel&reg; oneAPI DPC++/C++ Compiler <ul><li>Offloading compute intensive parts of the application using lambda kernel</li><li>Measuring kernel execution time</li></ul>
| Time to complete                  | 15 minutes


## Purpose
This sample uses blocked Floyd-Warshall all pairs shortest paths algorithm to compute a matrix that represents the minimum distance from any node to all other nodes in the graph. Using parallel blocked processing, blocks can be calculated simultaneously by distributing task computations to the GPU. For comparison, the application is run sequentially and parallelly with run times for each displayed in the application's output. The device where the code is run is also identified.

The parallel implementation of blocked Floyd-Warshall algorithm has three phases. Given a prior round of these computation phases are complete, phase 1 is independent; Phase 2 can only execute after phase 1 completes; Similarly phase 3 depends on phase 2 so can only execute after phase 2 is complete.

The inner loop of the sequential implementation is:
  g[i][j] = min(g[i][j], g[i][k] + g[k][j])

A careful observation shows that for the kth iteration of the outer loop, the computation depends on cells either on the kth column, g[i][k] or on the kth row, g[k][j] of the graph. Phase 1 handles g[k][k], phase 2 handles g[\*][k] and g[k][\*], and phase 3 handles g[\*][\*] in that sequence. This cell level observations largely propagate to the blocks as well. 

In each phase computation within a block can proceed independently in parallel. 


## Key implementation details
Includes device selector, unified shared memory, kernel, and command groups in order to implement a solution using parallel block method targeting the GPU. 


## License  
This code sample is licensed under MIT license. 


## Building the Program for CPU and GPU

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel&reg; oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### On a Linux* System

Perform the following steps:

1.  Build the program using the following <code> cmake </code> commands.
```
    $ cd all-pairs-shortest-paths
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

2.  Run the program <br>
```
    $ make run

```

### On a Windows* System Using Visual Studio* version 2017 or Newer

* Build the program using VS2017 or VS2019: Right click on the solution file and open using either VS2017 or VS2019 IDE. Right click on the project in Solution explorer and select Rebuild. From top menu select Debug -> Start without Debugging.
* Build the program using MSBuild: Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019". Run - MSBuild all-pairs-shortest-paths.sln /t:Rebuild /p:Configuration="Release"


## Running the sample

### Example Output
```
Device: Intel(R) Gen9
Repeating computation 8 times to measure run time ...
Iteration: 1
Iteration: 2
Iteration: 3
...
Iteration: 8
Successfully computed all pairs shortest paths in parallel!
Time sequential: 0.583029 sec
Time parallel: 0.159223 sec

```
