## Introduction

The oneAPI samples repository provides code samples for Intel oneAPI toolkits.<br><br>We recommend checking out a specific stable release version of the repository to [View available stable releases](https//github.com/oneapi-src/oneAPI-samples/tags). 
The latest versions (2021.4.0) of code samples on the master branch are not guaranteed to be stable.
 ### Sample Details

The oneAPI sample repository is organized as follows:

* [AI-and-Analytics:](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics)
  * [End-to-End-Workloads](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads)
  * [Features-and-Functionality](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality)
  * [Getting-Started-Samples](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples)
* [DirectProgramming](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming)
  * [C++](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2B)
  * [DPC++](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B)
  * [DPC++FPGA](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA)
    * [Reference Designs](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/ReferenceDesigns)
    * [Tutorials](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/Tutorials)
  * [Fortran](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA)
* [Libraries](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries)
  * [oneCCl](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneCCL)
  * [oneDAL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDAL)
  * [oneDNN](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDNN)
  * [oneDPL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDPL)
  * [oneMKL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneMKL)
  * [oneTBB](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneTBB)
  * [oneVPL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneVPL)
* [Publications](https://github.com/oneapi-src/oneAPI-samples/Publications/)
  * [Data Parallel C++](https://github.com/oneapi-src/oneAPI-samples/Publications/Data_Parallel_C%2B%2B)
* [Tools](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/)
  * [Advisor](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Advisor)
  * [Application Debugger](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/ApplicationDebugger)
  * [Benchmark](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Benchmark)
  * [IoT Connections Tools](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/IoTConnectionsTools)
  * [Migration](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration)
  * [Socwatch](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Socwatch)
  * [Trace](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Trace)
  * [UEFI debug](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/UEFI%20debug)
  * [VTune Profiler](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/VTuneProfiler)
)
To view an alphabetized list of all samples with descriptions by:
- Samples in Alphabetical order w/ device target](https://github.com/oneapi-src/CODESAMPLESLIST.md/)
- Samples by [Change History](https://github.com/oneapi-src/oneAPI-samples/CHANGELOGS.md)

### On Windows Platform

- If you are using Visual Studio 2019, Visual Studio 2019 version 16.4.0 or newer is required.
- Windows support for the FPGA code samples is limited to the FPGA emulator and optimization reports. Compile targets for FPGA hardware are provided on Linux only. See any FPGA code sample for more details.
- If you encounter a compilation error when building a sample program, such as the example error below, the directory path of the sample may be too long. The workaround is to move the sample to a temp directory.
- Example error: *Error MSB6003 The specified task executable dpcpp.exe could not be run .......

## Known Issues or Limitations

## Contributing

See [CONTRIBUTING wiki](https://github.com/oneapi-src/oneAPI-samples/blob/master/CONTRIBUTING.md) for more information.



## New Code Samples

|Version Introduced   |Sample Name|Description|
 |-----------------------|-------------------------------------------|---------------|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 01 - Introduction](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch01_intro)|fig_1_1_hello.cpp - Hello data-parallel programming <br> fig_1_3_race.cpp - Adding a race condition to illustrate a point about being asynchronous <br> fig_1_4_lambda.cpp - Lambda function in C++ code <br> fig_1_6_functor.cpp - Function object instead of a lambda (more on this in Chapter 10)|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 02 - Where Code Executes](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch02_where_code_runs)|fig_2_2_simple_program.cpp - Simple SYCL program  <br> fig_2_7_implicit_default_selector.cpp - Implicit default device selector through trivial construction of a queue  <br> fig_2_9_host_selector.cpp - Selecting the host device using the host_selector class <br> fig_2_10_cpu_selector.cpp - CPU device selector example <br> fig_2_12_multiple_selectors.cpp  - Example device identification output from various classes of device selectors and demonstration that device selectors can be used for cons <br> fig_2_13_gpu_plus_fpga.cpp - Creating queues to both GPU and FPGA devices  <br> fig_2_15_custom_selector.cpp - Custom selector for Intel Arria FPGA device  <br> fig_2_18_simple_device_code.cpp - Submission of device code  <br> fig_2_22_simple_device_code_2.cpp - Submission of device code  <br> fig_2_23_fallback.cpp - Fallback queue example|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 03 - Data Management](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch03_data_management)|fig_3_4_usm_explicit_data_movement.cpp - USM explicit data movement  <br> fig_3_5_usm_implicit_data_movement.cpp - USM implicit data movement  <br> fig_3_6_buffers_and_accessors.cpp - Buffers and accessors  <br> fig_3_10_in_order.cpp - In-order queue usage  <br> fig_3_11_depends_on.cpp - Using events and depends_on <br> fig_3_13_read_after_write.cpp - Read-after-Write  <br> fig_3_15_write_after_read_and_write_after_write.cpp - Write-after-Read and Write-after-Write |
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 04 - Expresssing Parallelism](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch04_expressing_parallelism)|fig_4_5_vector_add.cpp - Expressing a vector addition kernel with parallel_for <br> fig_4_6_matrix_add.cpp - Expressing a matrix addition kernel with parallel_for <br> fig_4_7_basic_matrix_multiply.cpp - Expressing a naïve matrix multiplication kernel for square matrices, with parallel_for <br> fig_4_13_nd_range_matrix_multiply.cpp - Expressing a naïve matrix multiplication kernel with ND-range parallel_for <br> fig_4_20_hierarchical_matrix_multiply.cpp - Expressing a naïve matrix multiplication kernel with hierarchical parallelism <br> fig_4_22_hierarchical_logical_matrix_multiply.cpp - Expressing a naïve matrix multiplication kernel with hierarchical parallelism and a logical range|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 05 - Error Handling](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch05_error_handling)|fig_5_1_async_task_graph.cpp - Separation of host program and task graph executions <br> fig_5_2_sync_error.cpp - Creating a synchronous error <br> fig_5_3_async_error.cpp - Creating an asynchronous error <br> fig_5_4_unhandled_exception.cpp - Unhandled exception in C++ <br> fig_5_5_terminate.cpp - std::terminate is called when a SYCL asynchronous exception isn’t handled <br> fig_5_6_catch_snip.cpp - Pattern to catch sycl::exception specifically <br> fig_5_7_catch.cpp - Pattern to catch exceptions from a block of code <br> fig_5_8_lambda_handler.cpp - Example asynchronous handler implementation defined as a lambda <br> fig_5_9_default_handler_proxy.cpp - Example of how the default asynchronous handler behaves |
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 06 - Unified Shared Memory](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch06_unified_shared_memory)|fig_6_5_allocation_styles.cpp - Three styles for allocation <br> fig_6_6_usm_explicit_data_movement.cpp - USM explicit data movement example <br> fig_6_7_usm_implicit_data_movement.cpp - USM implicit data movement example <br> fig_6_8_prefetch_memadvise.cpp - Fine-grained control via prefetch and mem_advise <br> fig_6_9_queries.cpp - Queries on USM pointers and devices |
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 07 - Buffers](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch07_buffers)|fig_7_2_3_4_creating_buffers.cpp - Creating buffers, Part 1 - Figure 7-3. Creating buffers, Part 2  - Figure 7-4. Creating buffers, Part 3 <br> fig_7_5_buffer_properties.cpp - Buffer properties <br> fig_7_8_accessors_simple.cpp - Simple accessor creation <br> fig_7_10_accessors.cpp - Accessor creation with specified usage|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 08 - Scheduling Kernals and Data Movement](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch08_graph_scheduling)|fig_8_3_linear_dependence_in_order.cpp - Linear dependence chain with in-order queues <br> fig_8_4_linear_dependence_events.cpp - Linear dependence chain with events <br> fig_8_5_linear_dependence_buffers.cpp - Linear dependence chain with buffers and accessors <br> fig_8_6_y_in_order.cpp - Y pattern with in-order queues <br> fig_8_7_y_events.cpp - Y pattern with events <br> fig_8_8_y_buffers.cpp - Y pattern with accessors|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 09 - Communication and Synchronization](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch09_work_item_communication)|fig_9_4_naive_matrix_multiplication.cpp - The naïve matrix multiplication kernel from Chapter 4 <br> fig_9_7_local_accessors.cpp - Declaring and using local accessors <br> fig_9_8_ndrange_tiled_matrix_multiplication.cpp - Expressing a tiled matrix multiplication kernel with an ND-range parallel_for and work-group local memory <br> fig_9_9_local_hierarchical.cpp - Hierarchical kernel with a local memory variable <br> fig_9_10_hierarchical_tiled_matrix_multiplication.cpp - A tiled matrix multiplication kernel implemented as a hierarchical kernel <br> fig_9_11_sub_group_barrier.cpp - Querying and using the sub_group class <br> fig_9_13_matrix_multiplication_broadcast.cpp - Matrix multiplication kernel includes a broadcast operation <br> fig_9_14_ndrange_sub_group_matrix_multiplication.cpp - Tiled matrix multiplication kernel expressed with ND-range parallel_for and sub-group collective functions|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 10 - Defining Kernels](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch10_expressing_kernels)|<br> fig_10_2_kernel_lambda.cpp - Kernel defined using a lambda expression <br> fig_10_3_optional_kernel_lambda_elements.cpp - More elements of a kernel lambda expression, including optional elements <br> fig_10_4_named_kernel_lambda.cpp - Naming kernel lambda expressions <br> fig_10_5_unnamed_kernel_lambda.cpp - Using unnamed kernel lambda expressions <br> fig_10_6_kernel_functor.cpp - Kernel as a named function object <br> fig_10_7_opencl_source_interop.cpp - Kernel created from OpenCL C kernel source <br> fig_10_8_opencl_object_interop.cpp - Kernel created from an OpenCL kernel object <br> fig_10_9_kernel_lambda_build_options.cpp - Compiling kernel lambdas with build options|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 11 - Vectors](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch11_vectors)|fig_11_6_load_store.cpp - Use of load and store member functions. <br> fig_11_7_swizzle_vec.cpp - Example of using the __swizzled_vec__ class <br> fig_11_8_vector_exec.cpp - Vector execution example |
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 12 - Device Information](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch12_device_information)|fig_12_1_assigned_device.cpp - Device we have been assigned by default <br> fig_12_2_try_catch.cpp - Using try-catch to select a GPU device if possible, host device if not <br> fig_12_3_device_selector.cpp - Custom device selector—our preferred solution <br> fig_12_4_curious.cpp - Simple use of device query mechanisms: curious.cpp <br> fig_12_6_very_curious.cpp - More detailed use of device query mechanisms: verycurious.cpp <br> fig_12_7_invocation_parameters.cpp - Fetching parameters that can be used to shape a kernel|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 13 - Practical Tips](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch13_practical_tips)|fig_13_4_stream.cpp - sycl::stream <br> fig_13_6_common_buffer_pattern.cpp - Common pattern—buffer creation from a host allocation <br> fig_13_7_common_pattern_bug.cpp - Common bug: Reading data directly from host allocation during buffer lifetime <br> fig_13_8_host_accessor.cpp - Recommendation: Use a host accessor to read kernel result <br> fig_13_9_host_accessor_for_init.cpp - Recommendation: Use host accessors for buffer initialization and reading of results <br> fig_13_10_host_accessor_deadlock.cpp - Bug (hang!) from improper use of host_accessors |
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 14 - Common Parallel Patterns](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch14_common_parallel_patterns)|fig_14_8_one_reduction.cpp - Reduction expressed as an ND-range data-parallel kernel using the reduction library <br> fig_14_11_user_defined_reduction.cpp - Using a user-defined reduction to find the location of the minimum value with an ND-range kernel <br> fig_14_13_map.cpp - Implementing the map pattern in a data-parallel kernel <br> fig_14_14_stencil.cpp - Implementing the stencil pattern in a data-parallel kernel <br> fig_14_15_local_stencil.cpp - Implementing the stencil pattern in an ND-range kernel, using work-group local memory <br> fig_14_18-20_inclusive_scan.cpp - Implementing a naïve reduction expressed as a  data-parallel kernel <br> fig_14_22_local_pack.cpp - Using a sub-group pack operation to build a list of elements needing additional postprocessing <br> fig_14_24_local_unpack.cpp - Using a sub-group unpack operation to improve load balancing for kernels with divergent control flow|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 15 - Programming for GPUs](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch15_gpus)|fig_15_3_single_task_matrix_multiplication.cpp - A single task matrix multiplication looks a lot like CPU host code <br> fig_15_5_somewhat_parallel_matrix_multiplication.cpp - Somewhat-parallel matrix multiplication <br> fig_15_7_more_parallel_matrix_multiplication.cpp - Even more parallel matrix multiplication <br> fig_15_10_divergent_control_flow.cpp - Kernel with divergent control flow <br> fig_15_12_small_work_group_matrix_multiplication.cpp - Inefficient single-item, somewhat-parallel matrix multiplication <br> fig_15_18_columns_matrix_multiplication.cpp - Computing columns of the result matrix in parallel, not rows|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 16 - Programming for CPUs](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch16_cpus)|fig_16_6_stream_triad.cpp - DPC++ STREAM Triad parallel_for kernel code <br> fig_16_12_forward_dep.cpp - Using a sub-group to vectorize a loop with a forward dependence <br> fig_16_18_vector_swizzle.cpp - Using vector types and swizzle operations in the single_task kernel|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 17 - Programming for FPGA](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch17_fpgas)|fig_17_9_fpga_selector.cpp - Choosing an FPGA device at runtime using the <br> fig_17_11_fpga_emulator_selector.cpp - Using fpga_emulator_selector for rapid development and debugging <br> fig_17_17_ndrange_func.cpp - Multiple work-item (16 × 16 × 16) invocation of a random number generator <br> fig_17_18_loop_func.cpp - Loop-carried data dependence (state) <br> fig_17_20_loop_carried_deps.cpp - Loop with two loop-carried dependences (i.e., i and a) <br> fig_17_22_loop_carried_state.cpp - Random number generator that depends on previous value generated <br> fig_17_31_inter_kernel_pipe.cpp - Pipe between two kernels: (1) ND-range and (2) single task with a loop  |
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 18 - Libraries](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch18_using_libs)|fig_18_1_builtin.cpp - Using std::log and sycl::log <br> fig_18_7_swap.cpp - Using std::swap in device code <br> fig_18_11_std_fill.cpp - Using std::fill <br> fig_18_13_binary_search.cpp - Using binary_search <br>fig_18_15_pstl_usm.cpp - Using Parallel STL with a USM allocator <br> Errata - code samples for 18-10, 18-12, 18-14, and 19-17 are not in the repository |
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 19 - Memory Model and Atomics](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Ch19_memory_model_and_atomics)|fig_19_3_data_race.cpp - Kernel containing a data race <br> fig_19_6_avoid_data_race_with_barrier.cpp - Avoiding a data race using a barrier <br>  fig_19_7_avoid_data_race_with_atomics.cpp - Avoiding a data race using atomic operations <br> fig_19_15_buffer_and_atomic_ref.cpp - Accessing a buffer via an explicitly created atomic_ref <br> fig_19_16_atomic_accessor.cpp - Accessing a buffer via an atomic_ref implicitly created by an atomic accessor <br> fig_19_18_histogram.cpp - Computing a histogram using atomic references in different memory spaces <br>  fig_19_19-20_device_latch.cpp - Combining Figure 19-20. Using and building a simple device-wide latch on top of atomic references <br> Errata - code samples for 18-10, 18-12, 18-14, and 19-17 are not in the repository|
|2021.4.0|[Pub: Data Parallel C++:](https://www.apress.com/9781484255735)

[Chapter 20 - Epilogue Future Direction](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC++/Epilogue_future_direction)|Epilogue source code examples: Future Direction of DPC++ <br> fig_ep_1_mdspan.cpp - Attaching accessor-like indexing to a USM pointer using mdspan <br> fig_ep_2-4_generic_space.cpp - Storing pointers to a specific address space in a class - Figure EP-3. Storing pointers to the generic address space in a class - Figure EP-4. Storing pointers with an optional address space in a class <br> fig_ep_5_extension_mechanism.cpp - Checking for Intel sub-group extension compiler support with #ifdef <br> fig_ep_6_device_constexpr.cpp - Specializing kernel code based on device aspects at kernel compile time <br> fig_ep_7_hierarchical_reduction.cpp - Using hierarchical parallelism for a hierarchical reduction|
|2021.4.0|[Merge Sort](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/ReferenceDesigns/merge_sort)|A Reference design demonstrating merge sort on an Intel® FPGA|
|2021.4.0|[Private Copies](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/Tutorials/Features/private_copies)|An Intel® FPGA tutorial demonstrating how to use the private_copies attribute to trade off the resource use and the throughput of a DPC++ FPGA program|
|2021.4.0|[Stall Enable](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/Tutorials/Features/stall_enable)|An Intel® FPGA tutorial demonstrating the use_stall_enable_clusters attribute|
## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
This doc was created on: August 25, 2021