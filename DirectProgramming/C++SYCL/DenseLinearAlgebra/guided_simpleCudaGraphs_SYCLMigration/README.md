# `Simple Cuda Graphs` Sample
 
The `SimpleCudaGraphs` sample demonstrates the migration of CUDA Graph explicit API calls to SYCL using the Taskflow programming model which manages a task dependency graph. This sample is implemented using SYCL* by migrating code from the original CUDA source code and offloading computations to a CPU, GPU, or accelerator.

| Area                      | Description
|:---                       |:---
| What you will learn       | Migrate and Map SYCL Taskflow equivalent of CUDA Graph API's
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

> **Note**: This sample is based on the [SimpleCudaGraphs](https://github.com/NVIDIA/cuda-samples/tree/v11.8/Samples/3_CUDA_Features/simpleCudaGraphs) sample in the NVIDIA/cuda-samples GitHub repository.

## Purpose

The sample shows the migration of simple explicit CUDA Graph APIs such as cudaGraphCreate, cudaGraphAddMemcpyNode, cudaGraphClone, etc, to SYCL equivalent APIs using [Taskflow](https://github.com/taskflow/taskflow) programming Model. The parallel implementation demonstrates the use of concepts, such as

- CUDA Graph APIs
- CUDA streams
- shared memory
- cooperative groups
- warp-level primitives. 

> **Note**: We use Intel® open-sources SYCLomatic tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. Users can also use SYCLomatic Tool which comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains the output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some code that is not migrated and has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated`            | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 22.04
| Hardware                   | Intel® Gen9 <br> Gen11 <br> Xeon CPU <br> Data Center GPU Max
| Software                   | SYCLomatic (Tag - 20230720) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2023.2.1

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

Reduction is performed in two CUDA kernels `reduce ()` and `reduceFinal()`. We only migrate the cudaGraphsManual() using SYCLomatic Tool and manually migrate the code section which is not migrated using [Taskflow](https://github.com/taskflow/taskflow) Programming Model. We do not migrate `cudaGraphsUsingStreamCapture()` because CUDA Stream Capture APIs are not yet supported in SYCL.

> **Note**: For more information on how to use Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.vmhplg).

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

4. Pass the JSON file as input to the SYCLomatic tool. The result is written to a folder named dpct_output. The `--in-root` specifies the path to the root of the source tree to be migrated. The `--use-custom-helper` option will make a copy of dpct header files/functions used in migrated code into the dpct_output folder as `include` folder. The `--use-experimental-features` option specifies an experimental helper function used to logically group work-items.
   ```
   c2s -p compile_commands.json --in-root ../../.. --use-custom-helper=api --use-experimental-features=logical-group
   ```

### Manual workarounds 

The following warnings in the "DPCT1XXX" format are generated by the tool to indicate the code has not been migrated by the tool and needs to be manually modified in order to complete the migration. 

1.	DPCT1007: Migration of cudaGraphCreate is not supported.
    ```
    cudaGraphCreate(&graph, 0);
    ```
    SYCL doesn’t support migration of CUDA Graphs API yet. We can manually migrate these APIs with the help of [Taskflow](https://github.com/taskflow/taskflow) programming model which supports SYCL. Taskflow introduces a lightweight task graph-based programming model, [tf::syclFlow](https://github.com/taskflow/taskflow/tree/master/taskflow/sycl), for tasking SYCL operations and their dependencies. We need to include the header file, taskflow/sycl/syclflow.hpp, for using tf::syclFlow.
    ```
    tf::Taskflow tflow;
    tf::Executor exe;
    ```
    The above code lines construct a taskflow and an executor. The graph created by the taskflow is executed by an executor. 
   
2.	DPCT1007: Migration of cudaGraphAddMemcpyNode is not supported.
    ```
    cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
    ```
    The tf::syclFlow provides memcpy method to creates a memcpy task that copies untyped data in bytes.
    ```
    tf::syclTask inputVec_h2d = sf.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize) .name("inputVec_h2d");
    ```  
3.	DPCT1007: Migration of cudaGraphAddMemsetNode is not supported.
    ```
    cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);
    ```
    The tf::syclFlow::memset method creates a memset task that fills untyped data with a byte value. 
    ```
     tf::syclTask outputVec_memset = sf.memset(outputVec_d, 0, numOfBlocks * sizeof(double)) .name("outputVecd_memset");
    ```
    For more information on memory operations refer [here](https://github.com/taskflow/taskflow/blob/master/taskflow/sycl/syclflow.hpp).

4.	DPCT1007: Migration of cudaGraphAddKernelNode is not supported.
    ```
    cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams);
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
5.	DPCT1007: Migration of cudaGraphAddHostNode is not supported.
    ```
    cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(),   nodeDependencies.size(), &hostParams);
    ```
    The tf::syclFlow doesn’t have a host method to run the callable on the host, instead we can achieve this by creating a subflow graph since Taskflow supports dynamic tasking and runs the callable on the host.
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
    CUDA graph nodes are equivalent to SYCL tasks, both tf::Taskflow and tf::syclFlow class include num_tasks() function to query the total number of tasks.
    ```
    sf_Task = sf.num_tasks();
    ```
7.	DPCT1007: Migration of cudaGraphInstantiate is not supported.
    ```
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    ```
    SYCL Task graph doesn’t need to be instantiated before executing but need to establish the task dependencies using precede and succeed.
    ```
    reduce_kernel.succeed(inputVec_h2d, outputVec_memset).precede(reduceFinal_kernel);
    reduceFinal_kernel.succeed(resultd_memset).precede(result_d2h);
    ```
    The inputVec_h2d and outputVec_memset tasks run parallelly followed by reduce_kernel task.
   
8. DPCT1007: Migration of cudaGraphClone is not supported.
    ```
    cudaGraphClone(&clonedGraph, graph);
    ```
    In SYCL, there is no clone function available as Taskflow graph objects are move-only. To achieve functionality, we can use std::move() function as shown below.
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
    A taskflow graph can be run once or multiple times using an executor. run_n() will run the taskflow the number of times specified by the second argument.
    ```
    exe.run_n(tflow, GRAPH_LAUNCH_ITERATIONS).wait();
    ```   
10. DPCT1007: Migration of cudaGraphExecDestroy is not supported.
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

## Build and Run the `Simple CUDA Graphs` Sample

>  **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
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
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run
    $ unset ONEAPI_DEVICE_SELECTOR
    ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.
  
## Example Output

The following example is for `02_sycl_migrated` for GPU on **Intel(R) UHD Graphics P630 [0x3e96]**.
```
16777216 elements
threads per block  = 256
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

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
