# `Simple Cuda Graphs` Sample
 
The simpleCudaGraphs sample demonstrates the migration of CUDA Graph explicit API calls to SYCL using Taskflow programming model which manages a task dependency graph. This sample is implemented using SYCL* by migrating code from original CUDA source code and offloading computations to a GPU/CPU.

| Property                  | Description
|:---                       |:---
| What you will learn       | Migrate and Map of SYCL Taskflow equivalent of CUDA Graph API's
| Time to complete          | 15 minutes

## Purpose

The sample shows the migration of simple explicit CUDA Graph API's such as cudaGraphCreate, cudaGraphAddMemcpyNode, cudaGraphClone etc, to SYCL equivalent API's using [Taskflow](https://github.com/taskflow/taskflow) programming Model. The parallel implementation demonstrates the use of CUDA Graph API's, CUDA streams, shared memory, cooperative groups and warp level primitives. 

> **Note**: We use Intel® open-sources SYCLomatic tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. User's can also use SYCLomatic Tool which comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated`            | Contains manually migrated SYCL code from CUDA code.

### Workflow For CUDA to SYCL migration

Refer [Workflow](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for details.

### CUDA source code evaluation

The simpleCudaGraphs sample demonstrates the usage of CUDA Graphs API’s by performing element reduction. The CUDA Graph API are demonstrated in two CUDA functions cudaGraphsManual() which uses explicit CUDA Graph APIs and cudaGraphsUsingStreamCapture() which uses stream capture APIs. Reduction is performed in two CUDA kernels reduce () and reduceFinal(). We only migrate the cudaGraphsManual() using SYCLomatic Tool and manually migrating the unmigrated code section using [Taskflow](https://github.com/taskflow/taskflow) Programming Model. We do not migrate cudaGraphsUsingStreamCapture() because CUDA Stream Capture APIs are not yet supported in SYCL.

This sample is migrated from NVIDIA CUDA sample. See the [SimpleCudaGraphs](https://github.com/NVIDIA/cuda-samples/tree/v11.8/Samples/3_CUDA_Features/simpleCudaGraphs) sample in the NVIDIA/cuda-samples GitHub.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 20.04
| Hardware                   | Intel® Gen9, Gen11 and Xeon CPU
| Software                   | SYCLomatic version 2023.0, Intel oneAPI Base Toolkit version 2023.0

For more information on how to use Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features: 
- CUDA Graph APIs
- CUDA Stream Capture
- Shared memory
- CUDA streams 
- Cooperative groups
- Warp level primitives

SYCL simpleCudaGraphs sample performs reduction operarion to obtain the sum value from 16777216 number of elements in two different computational kernels reduce and reduceFinal. These kernels are scheduled through taskflow which develops a simple and powerful task programming model to enable efficient implementations of heterogeneous decomposition strategies and leverages both static and dynamic task graph constructions to incorporate computational patterns.

## Build the `simpleCudaGraphs` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Tool assisted migration – SYCLomatic 

For this sample, the SYCLomatic Tool automatically migrates ~80% of the CUDA runtime API's to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. git clone https://github.com/NVIDIA/cuda-samples.git
2. cd cuda-samples/Samples/3_CUDA_Features/simpleCudaGraphs/
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
4. The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.
5. Pass the JSON file as input to the SYCLomatic Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --use-custom-helper=api
   ```
   
### Manual workarounds 

The following warnings in the "DPCT1XXX" format are gentereated by the tool to indicate the code not migrated by the tool and need to be manually modified in order to complete the migration. 

1. DPCT1007: Migration of size is not supported.
   ```
   for (int i = 0; i < cta.size(); i += tile32.size()) {
   ```
   SYCLomatic tool migrates the CUDA thread-block to SYCL group class but doesn’t migrate its member function call size(). We have to manually change it to get_local_linear_range() member function which returns the total number of work-items in the workgroup.
   ```
   for (int i = 0; i < cta.get_local_linear_range(); i += tile32.size()) {
   ```

2.	DPCT1007: Migration of shfl_down is not supported.
    ```
    temp_sum += tile32.shfl_down(temp_sum, offset);
    ```
    We need to manually change the syntax to shuffle_down to get it functionally working. Following is the workaround for the above code:
    ```
    temp_sum += tile32.shuffle_down(temp_sum, offset);
    ```
3.	DPCT1007: Migration of cudaGraphCreate is not supported.
    ```
    cudaGraphCreate(&graph, 0);
    ```
    SYCL doesn’t support migration of CUDA Graphs API yet. We can manually migrate these APIs with the help of [Taskflow](https://github.com/taskflow/taskflow) programming model which supports SYCL. Taskflow introduces a lightweight task graph-based programming model, [tf::syclFlow](https://github.com/taskflow/taskflow/tree/master/taskflow/sycl), for tasking SYCL operations and their dependencies. We need to include the header file, taskflow/sycl/syclflow.hpp, for using tf::syclFlow.
    ```
    tf::Taskflow tflow;
    tf::Executor exe;
    ```
    The above code lines construct a taskflow and an executor. The graph created by the taskflow is executed by an executor. 
   
4.	DPCT1007: Migration of cudaGraphAddMemcpyNode is not supported.
    ```
    cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
    ```
    The tf::syclFlow provides memcpy method to creates a memcpy task that copies untyped data in bytes.
    ```
    tf::syclTask inputVec_h2d = sf.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize) .name("inputVec_h2d");
    ```  
5.	DPCT1007: Migration of cudaGraphAddMemsetNode is not supported.
    ```
    cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
    ```
    The tf::syclFlow::memset method creates a memset task that fills untyped data with a byte value. 
    ```
     tf::syclTask outputVec_memset = sf.memset(outputVec_d, 0, numOfBlocks * sizeof(double)) .name("outputVecd_memset");
    ```
    For more information on memory operations refer [here](https://github.com/taskflow/taskflow/blob/master/taskflow/sycl/syclflow.hpp).

6.	DPCT1007: Migration of cudaGraphAddKernelNode is not supported.
    ```
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));
    ```
    The tf::syclFlow::on creates a task to launch the given command group function object and tf::syclFlow::parallel_for creates a kernel task from a parallel_for method through the handler object associated with a command group. The SYCL runtime schedules command group function objects from out-of-order queue and constructs a task graph based on submitted events.
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
7.	DPCT1007: Migration of cudaGraphAddHostNode is not supported.
    ```
    cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(),   nodeDependencies.size(), &hostParams));
    ```
    The tf::syclFlow doesn’t have a host method to run the callable on the host, instead we can achieve this by creating a subflow graph since Taskflow supports dynamic tasking and runs the callable on the host.
    ```
    tf::Task syclHostTask = tflow.emplace([&](){
      myHostNodeCallback(&hostFnData);
    }).name("syclHostTask");
    syclHostTask.succeed(syclKernelTask);   
    ```
    The task dependencies are established through precede or succeed, here syclHostTask runs after syclKernelTask.
   
8.	DPCT1007: Migration of cudaGraphGetNodes is not supported.
    ```
    cudaGraphGetNodes(graph, nodes, &numNodes));
    ```
    CUDA graph nodes are equivalent to SYCL tasks, both tf::Taskflow and tf::syclFlow class include num_tasks() function to query the total number of tasks.
    ```
    sf_Task = sf.num_tasks();
    ```
9.	DPCT1007: Migration of cudaGraphInstantiate is not supported.
    ```
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    ```
    SYCL Task graph doesn’t need to be instantiated before executing but need to establish the task dependencies using precede and succeed.
    ```
    reduce_kernel.succeed(inputVec_h2d, outputVec_memset).precede(reduceFinal_kernel);
    reduceFinal_kernel.succeed(resultd_memset).precede(result_d2h);
    ```
    The inputVec_h2d and outputVec_memset tasks run parallelly followed by reduce_kernel task.
   
10. DPCT1007: Migration of cudaGraphClone is not supported.
    ```
    cudaGraphClone(&clonedGraph, graph));
    ```
    In SYCL, there is no clone function available as Taskflow graph objects are move-only. To achieve functionality, we can use std::move() function as shown below.
    ```
    tf::Taskflow tflow_clone(std::move(tflow));
    ```
    This will construct a taskflow tflow_clone from moved taskflow tflow, and taskflow tflow becomes empty. For more information refer [here](https://taskflow.github.io/taskflow/classtf_1_1Taskflow.html#afd790de6db6d16ddf4729967c1edebb5). 
    
11. DPCT1007: Migration of cudaGraphLaunch is not supported.
    ```
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
      cudaGraphLaunch(graphExec, streamForGraph);
    }
    ```
    A taskflow graph can be run once or multiple times using an executor. run_n() will run the taskflow the number of times specified by the second argument.
    ```
    exe.run_n(tflow, GRAPH_LAUNCH_ITERATIONS).wait();
    ```   
12. DPCT1007: Migration of cudaGraphExecDestroy is not supported.
    DPCT1007: Migration of cudaGraphDestroy is not supported.
    ```
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    ```
    tf::Taskflow class has default destructor operators for both tf::executor and tf::taskflow objects created.
    ```
    ~Executor() 
    ~Taskflow()
    ```
    To ensure that all the taskflow submissions are completed before calling destructor, we must use wait() during the execution.

> **Note**: The SYCL Task Graph Programming Model, syclFlow, leverages the out-of-order property of the SYCL queue to design a simple and efficient scheduling algorithm using topological sort. SYCL can be slower than CUDA graphs because of execution overheads.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence will build the `02_sycl_migrated` versions of the program.
   
3. Run the program.
   
   Run `02_sycl_migrated` on GPU.
   ```
   $ make run
   ```   
   Run `02_sycl_migrated` for CPU.
    ```
    export SYCL_DEVICE_FILTER=cpu
    make run
    unset SYCL_DEVICE_FILTER
    ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.
  
## Example Output

The following example is for `02_sycl_migrated` for GPU on **Intel(R) UHD Graphics [0x9a60]**.
```
16777216 elements
threads per block  = 512
Graph Launch iterations = 3
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214

Number of tasks(nodes) in the syclTaskFlow(graph) created manually = 7
Cloned Graph Output.. 
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
Built target run_gpu
```
>**Note**: On Gen11 architecture double data types are not supported, hence change double data types to float data types.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

