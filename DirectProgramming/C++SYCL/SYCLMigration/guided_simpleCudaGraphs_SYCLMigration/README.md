# `Simple Cuda Graphs` Sample
 
The `SimpleCudaGraphs` sample demonstrates the migration of CUDA Graph explicit API calls to SYCL using two separate approaches,
1. The Taskflow programming model which manages a task dependency graph
2. The SYCL Graph extension with command groups SYCL creates an implicit dependency graph of kernel execution at runtime

The original CUDA* source code is migrated to SYCL for portability across GPUs from multiple vendors.

| Area                      | Description
|:---                       |:---
| What you will learn       | Migrate and Map SYCL equivalent of CUDA Graph API's using two approaches 
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

> **Note**: This sample is based on the [SimpleCudaGraphs](https://github.com/NVIDIA/cuda-samples/tree/v11.8/Samples/3_CUDA_Features/simpleCudaGraphs) sample in the NVIDIA/cuda-samples GitHub repository.

## Purpose

The sample shows the migration of simple explicit CUDA Graph APIs such as cudaGraphCreate, cudaGraphAddMemcpyNode, cudaGraphClone, etc, to SYCL equivalent APIs using [Taskflow](https://github.com/taskflow/taskflow) programming Model and SYCL [Graph](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc#sycl_ext_oneapi_graph) extension. The parallel implementation demonstrates the use of concepts, such as

- CUDA Graph APIs
- CUDA streams
- shared memory
- cooperative groups
- warp-level primitives. 

> **Note**: The sample used the open-source SYCLomatic tool that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the Intel® DPC++ Compatibility Tool available to augment the Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                           | Description
|:---                                   |:---
| `01_dpct_output`                      | Contains the output of the SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some code that is not migrated and has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated_option1`            | Contains manually migrated SYCL code from CUDA code using Taskflow programming model.
| `02_sycl_migrated_option2`            | Contains manually migrated SYCL code from CUDA code using SYCL Graph extension.

## Prerequisites

| Optimized for                                      | Description
|:---                                                |:---
| OS                                                 | Ubuntu* 22.04
| Hardware (for 02_sycl_migrated__option1)           | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Xeon CPU <br> Intel® Data Center GPU Max <br> Nvidia Testla P100 <br> Nvidia A100 <br> Nvidia H100
| Hardware (for 02_sycl_migrated__option2)           | Intel® Data Center GPU Max <br> 
| Software                                           | SYCLomatic (Tag - 20240403) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2024.1 <br> oneAPI for NVIDIA GPUs plugin (version 2024.1) from Codeplay

For more information on how to install Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) <br>
Refer to [oneAPI for NVIDIA GPUs plugin](https://developer.codeplay.com/products/oneapi/nvidia/) from Codeplay to execute a sample on NVIDIA GPU.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features: 

- CUDA Graph API
- CUDA Stream Capture
- Shared memory
- CUDA streams 
- Cooperative groups
- Warp-level primitives

SYCL simpleCudaGraphs sample performs reduction operations to obtain the sum value from 16777216 elements in two different computational kernels `reduce` and `reduceFinal`. These kernels are scheduled through taskflow which develops a simple and powerful task programming model to enable efficient implementations of heterogeneous decomposition strategies and leverages both static and dynamic task graph constructions to incorporate computational patterns.

>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.

### CUDA source code evaluation

The SimpleCudaGraphs sample demonstrates the usage of CUDA Graphs APIs by performing element reduction. 

The CUDA Graph API is demonstrated in two CUDA functions: 
1. `cudaGraphsManual()` which uses explicit CUDA Graph APIs 
2. `cudaGraphsUsingStreamCapture()` which uses stream capture API. 

Reduction is performed in two CUDA kernels `reduce ()` and `reduceFinal()`. 

In the first approach(`02_sycl_migrated__option1`) we only migrate the cudaGraphsManual() method using the SYCLomatic Tool and manually migrate the code section which is not migrated by the tool using [Taskflow](https://github.com/taskflow/taskflow) Programming Model. We do not migrate `cudaGraphsUsingStreamCapture()` because CUDA Stream Capture APIs are not yet supported in SYCL.

In the second approach(`02_sycl_migrated__option2`) we migrate using the SYCLomatic Tool and manually migrate the code section which is not migrated by the tool using SYCL [Graph](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc#sycl_ext_oneapi_graph) extension. The method cudaGraphsManual() is migrated using Explicit graph building API and the method cudaGraphsUsingStreamCapture() is migrated using Queue recording API.

> **Note**: For more information on how to use the Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `Simple CUDA Graphs` Code

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates ~80% of the CUDA runtime APIs to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the SimpleCudaGraphs sample directory.
   ```
   cd cuda-samples/Samples/3_CUDA_Features/simpleCudaGraphs/
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the SYCLomatic tool. The result is written to a folder named dpct_output. The `--in-root` specifies the path to the root of the source tree to be migrated. The `--gen-helper-function ` option will make a copy of the dpct header files/functions used in migrated code into the dpct_output folder as `include` folder. The `--use-experimental-features` option specifies an experimental helper function used to logically group work-items.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function --use-experimental-features=logical-group
   ```

### Manual workarounds 

The following warnings in the "DPCT1XXX" format are generated by the tool to indicate the code has not been migrated by the tool and needs to be manually modified to complete the migration. 
Below are the manual workarounds, Option 1 for 02_sycl_migrated__option1 and Option 2 for 02_sycl_migrated__option2 respectively.

1.	DPCT1007: Migration of cudaGraphCreate is not supported.
    ```
    cudaGraphCreate(&graph, 0);
    ```
    Option 1: SYCL doesn’t support migration of CUDA Graphs API yet. We can manually migrate these APIs with the help of [Taskflow](https://github.com/taskflow/taskflow) programming model which supports SYCL. Taskflow introduces a lightweight task graph-based programming model, [tf::syclFlow](https://github.com/taskflow/taskflow/tree/master/taskflow/sycl), for tasking SYCL operations and their dependencies. We need to include the header file, taskflow/sycl/syclflow.hpp, for using tf::syclFlow.
    ```
    tf::Taskflow tflow;
    tf::Executor exe;
    ```
    The above code lines construct a taskflow and an executor. The graph created by the taskflow is executed by an executor.
  	
  	 Option 2: SYCL Graph is an addition in `ext::oneapi::experimental` namespace, SYCL command_graph creates an object in the modifiable state for context `syclContext` and device `syclDevice`.

  	 ```
    namespace sycl_ext = sycl::ext::oneapi::experimental;
    sycl_ext::command_graph graph(q.get_context(), q.get_device());
    ```
   
2.	DPCT1007: Migration of cudaGraphAddMemcpyNode is not supported.
    ```
    cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
    ```
    Option 1: The tf::syclFlow provides memcpy method to create a memcpy task that copies untyped data in bytes.
    ```
    tf::syclTask inputVec_h2d = sf.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize) .name("inputVec_h2d");
    ```
    Option 2: Command graph class includes `add` method which creates an empty node that contains no command. Its intended use is to make a connection point inside a graph between groups of nodes, and can significantly reduce the number of edges, using this we can add memcpy operation as a node.
  	 ```
    auto nodecpy = graph.add([&](sycl::handler& h){
      h.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);
    }); 
    ```
3.	DPCT1007: Migration of cudaGraphAddMemsetNode is not supported.
    ```
    cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);
    ```
    Option 1: The tf::syclFlow::memset method creates a memset task that fills untyped data with a byte value. 
    ```
     tf::syclTask outputVec_memset = sf.memset(outputVec_d, 0, numOfBlocks * sizeof(double)) .name("outputVecd_memset");
    ```
    For more information on memory operations refer [here](https://github.com/taskflow/taskflow/blob/master/taskflow/sycl/syclflow.hpp).

  	Option 2: Similar to memcpy node, memset operation can also be included as a node through the command graph `add` method.
  	 ```
    auto nodememset1 = graph.add([&](sycl::handler& h){
      h.fill(outputVec_d, 0, numOfBlocks);
    });
    ```

4.	DPCT1007: Migration of cudaGraphAddKernelNode is not supported.
    ```
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams);
    ```
    Option 1: The tf::syclFlow::on creates a task to launch the given command group function object and tf::syclFlow::parallel_for creates a kernel task from a parallel_for method through the handler object associated with a command group. The SYCL runtime schedules command group function objects from an out-of-order queue and constructs a task graph based on submitted events.
    ```
    tf::syclTask reduce_kernel = sf.on([=] (sycl::handler& cgh){
      sycl::local_accessor<double, 1> tmp(sycl::range<1>(THREADS_PER_BLOCK), cgh);
      cgh.parallel_for(sycl::nd_range<3>{sycl::range<3>(1, 1, numOfBlocks) *
                                sycl::range<3>(1, 1, THREADS_PER_BLOCK), sycl::range<3>(1, 1, THREADS_PER_BLOCK)}, [=](sycl::nd_item<3> item_ct1)[[intel::reqd_sub_group_size(SUB_GRP_SIZE)]]
                      {
                        reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1, tmp.get_pointer());
                      });
        }).name("reduce_kernel");
    ```

    Option 2: Kernel operations are also included as a node through the command graph `add` method. These commands are captured into the graph and executed asynchronously when the graph is submitted to a queue. The `property::node::depends_on` property can be passed here with a list of nodes to create dependency edges on.
  	 ```
    auto nodek1 = graph.add([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1,
               tmp_acc_ct1.get_pointer());
      });
    },  sycl_ext::property::node::depends_on(nodecpy, nodememset1));
    ```
5.	DPCT1007: Migration of cudaGraphAddHostNode is not supported.
    ```
    cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(),   nodeDependencies.size(), &hostParams);
    ```
    Option 1: The tf::syclFlow doesn’t have a host method to run the callable on the host, instead, we can achieve this by creating a subflow graph since Taskflow supports dynamic tasking and runs the callable on the host.
    ```
    tf::Task syclHostTask = tflow.emplace([&](){
      myHostNodeCallback(&hostFnData);
    }).name("syclHostTask");
    syclHostTask.succeed(syclKernelTask);   
    ```
    The task dependencies are established through precede or succeed, here syclHostTask runs after syclKernelTask.
   
6.	DPCT1007: Migration of cudaGraphGetNodes is not supported.
    ```
    cudaGraphGetNodes(graph, nodes, &numNodes);
    ```
    Option 1: CUDA graph nodes are equivalent to SYCL tasks, both tf::Taskflow and tf::syclFlow classes include num_tasks() function to query the total number of tasks.
    ```
    sf_Task = sf.num_tasks();
    ```

7.	DPCT1007: Migration of cudaGraphInstantiate is not supported.
    ```
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    ```
    Option 1: SYCL Task graph doesn’t need to be instantiated before executing but needs to establish the task dependencies using precede and succeed.
    ```
    reduce_kernel.succeed(inputVec_h2d, outputVec_memset).precede(reduceFinal_kernel);
    reduceFinal_kernel.succeed(resultd_memset).precede(result_d2h);
    ```
    The inputVec_h2d and outputVec_memset tasks run parallelly followed by the reduce_kernel task.

  	Option 2: After all the operations are added as a node the graph is finalized using `finalize()` so that no more nodes can be added and creates an executable graph that 
    can be submitted for execution.
  	 ```
    auto exec_graph = graph.finalize();
    sycl::queue qexec = sycl::queue{sycl::gpu_selector_v, 
      {sycl::ext::intel::property::queue::no_immediate_command_list()}};
    
    ```
   
8. DPCT1007: Migration of cudaGraphClone is not supported.
    ```
    cudaGraphClone(&clonedGraph, graph);
    ```
    Option 1: In SYCL, there is no clone function available as Taskflow graph objects are move-only. To achieve functionality, we can use the std::move() function as shown below.
    ```
    tf::Taskflow tflow_clone(std::move(tflow));
    ```
    This will construct a taskflow tflow_clone from moved taskflow tflow, and taskflow tflow becomes empty. For more information refer [here](https://taskflow.github.io/taskflow/classtf_1_1Taskflow.html#afd790de6db6d16ddf4729967c1edebb5).
    
9. DPCT1007: Migration of cudaGraphLaunch is not supported.
    ```
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
      cudaGraphLaunch(graphExec, streamForGraph);
    }
    ```
    Option 1: A taskflow graph can be run once or multiple times using an executor. run_n() will run the taskflow the number of times specified by the second argument.
    ```
    exe.run_n(tflow, GRAPH_LAUNCH_ITERATIONS).wait();
    ```
    Option 2: The graph is submitted in its entirety for execution via `handler::ext_oneapi_graph(graph)`.
  	 ```
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    qexec.submit([&](sycl::handler& cgh) {
      cgh.ext_oneapi_graph(exec_graph);
    }).wait();
    }
    ```
10. DPCT1007: Migration of cudaGraphExecDestroy is not supported.
    DPCT1007: Migration of cudaGraphDestroy is not supported.
    ```
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    ```
    Option 1: tf::Taskflow class has default destructor operators for both tf::executor and tf::taskflow objects created.
    ```
    ~Executor() 
    ~Taskflow()
    ```
    To ensure that all the taskflow submissions are completed before calling the destructor, we must use wait() during the execution.

> **Note**: The SYCL Task Graph Programming Model, syclFlow, leverages the out-of-order property of the SYCL queue to design a simple and efficient scheduling algorithm using topological sort. SYCL can be slower than CUDA graphs because of execution overheads. Hence we prefer migrating with SYCL Graph Extension.


Below are the manual migration using SYCL graph extension for cudaGraphsUsingStreamCapture() method.


11. DPCT1027: The call to cudaStreamBeginCapture was replaced with 0 because SYCL currently does not support capture operations on queues.
```
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
    cudaStreamEndCapture(stream1, &graph);
```
The Queue Recording API (Record & Replay) captures command-groups submitted to a queue and records them in a graph. The command_graph::begin_recording and command_graph::end_recording entry-points return a bool value informing the user whether a related queue state change occurred. All the operation are placed in between these queue-recording APIs.

```
    sycl_ext::command_graph graph(q.get_context(), q.get_device());
    graph.begin_recording(q);
    ...
    graph.end_recording();
```


12. The memcpy, memset, and kernel operations are placed as a node via `sycl::event` namespace as follows,
```
     sycl::event ememcpy = q.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);

     sycl::event ememset = q.fill(outputVec_d, 0, numOfBlocks);

     sycl::event ek1 = q.submit([&](sycl::handler &cgh) {
     cgh.depends_on({ememcpy, ememset});
     sycl::local_accessor<double, 1> tmp_acc_ct1(
       sycl::range<1>(THREADS_PER_BLOCK), cgh);

     cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1,
               tmp_acc_ct1.get_pointer());
      });
   });
```


13.   DPCT1007: Migration of cudaGraphInstantiate is not supported.

```
   cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0);
```
Similar to Graph explicit API calls, After all the operations are added as a node the graph is finalized using `finalize()` so that no more nodes can be added and creates an executable graph that can be submitted for execution.
```
   auto exec_graph = graph.finalize();
   sycl::queue qexec = sycl::queue{sycl::gpu_selector_v, 
      {sycl::ext::intel::property::queue::no_immediate_command_list()}};
```


14.  DPCT1007:Migration of cudaGraphLaunch is not supported.
```
   cudaGraphLaunch(clonedGraphExec, streamForGraph);
```
The graph is then submitted in its entirety for execution via `handler::ext_oneapi_graph(graph)`.
```
   for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
      qexec.submit([&](sycl::handler& cgh) {
        cgh.ext_oneapi_graph(exec_graph);
      }).wait();
  }
```


15. CUDA code includes a custom API `findCUDADevice` in helper_cuda file to find the best CUDA Device available.
```
    findCudaDevice (argc, (const char **) argv);
```
Since it is a custom API SYCLomatic tool will not act on it and we can either remove it or replace it with the `sycl get_device()` API


## Build and Run the `Simple CUDA Graphs` Sample

>  **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script at the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*


### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```

   **Note**: By default, no flag are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU. <br>
    Enable **INTEL_MAX_GPU** flag during the build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performance. <br>
    Enable **NVIDIA_GPU** flag during the build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin from Codeplay is required to build for NVIDIA GPUs) <br>

   By default, this command sequence will build the `02_sycl_migrated__option1` and `02_sycl_migrated__option2` versions of the program.
   
3. Run the program.
   
   Run `02_sycl_migrated_option1` on GPU.
   ```
   $ make run_op1
   ```   
   Run `02_sycl_migrated_option1` for CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run_op1
    $ unset ONEAPI_DEVICE_SELECTOR
    ```
    Run `02_sycl_migrated_option2` on GPU(Intel® Data Center GPU Max Device).
   ```
   $ make run_op2
   ``` 

#### Troubleshooting

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-1/overview.html) for more information on using the utility.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
