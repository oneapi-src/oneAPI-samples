# `Guided Matrix Multiplication Bad Buffers` Sample

The `Guided Matrix Multiplication Bad Buffers` sample demonstrates how to use the Intel® oneAPI Base Toolkit (Base Kit) and several tools found in it to triage incorrect use of the SYCL language.

The sample is simple program that multiplies together two large matrices and verifies the results.

| Area                  | Description
|:---                   |:---
| What you will learn   | A method to determine the root cause problems from passing bad buffers through the SYCL runtime.
| Time to complete      | 50 minutes

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The two samples in this tutorial show examples of how to debug issues arising from passing bad buffers through the SYCL runtime, one when using SYCL buffers and one when using a direct reference to device memory.

In one case, we will know that there is a problem due to a crash. In the other case, we will get bad results.

The sample includes different versions of a simple matrix multiplication program.

| File                          | Description
|:---                           |:---
| `a1_matrix_mul_zero_buff.cpp` | This example shows the crash you get when a zero-element buffer is passed to a SYCL `submit`  lambda function.
| `a2_matrix_mul.cpp`           | A working version of the matrix multiply code that uses SYCL buffers and accessors.
| `b1_matrix_mul_null_usm.cpp`  | This example shows you the bad results you get when a null pointer to device memory is passed to a SYCL `submit` lambda function.
| `b2_matrix_mul_usm.cpp`       | A working version of the matrix multiply code that uses explicit pointers to host and device memory rather than SYCL buffers and accessors.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04
| Hardware                | GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler <br> Intel® Distribution for GDB* <br> [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace), which is available from the [onetrace](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) GitHub repository.

## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each workgroup)
- Explicit memory operations using Unified Shared Memory (USM)
- SYCL* kernels (including parallel_for function and explicit memory copies)
- SYCL* queues

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Guided Matrix Multiplication Bad Buffers` Programs

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
*[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the programs.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
3. Run the programs.
   ```
   make run_all
   ```
   >**Note**: The application by default uses the Level Zero runtime and will run without errors.  We will do a deeper investigation of the application, in particular with the openCL runtime, to expose problems that could also manifest in Level Zero.

   For the zero buffer version only, enter the following:
   ```
   make run_a1_zero_buff
   ```
   For the working buffers version only, enter the following:
   ```
   make run_a2
   ```
   For the USM null pointer version only, enter the following:
   ```
   make run_b1_null_usm
   ```
   For the working USM version only, enter the following:
   ```
   make run_b2_usm
   ```
4. Clean the program. (Optional)
   ```
   make clean
   ```


#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.


### Build and Run the Sample in Intel® DevCloud (Optional)

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. 

Use the Linux instructions to build and run the program.

You can specify a GPU node using a single line script.

```
qsub  -I  -l nodes=1:gpu:ppn=2 -d .
```
- `-I` (upper case I) requests an interactive session.
- `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
- `-d .` makes the current folder as the working directory for the task.

  |Available Nodes    |Command Options
  |:---               |:---
  |GPU                |`qsub -l nodes=1:gpu:ppn=2 -d .`
  |CPU                |`qsub -l nodes=1:xeon:ppn=2 -d .`

For more information on how to specify compute nodes read *[Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/)* in the Intel® DevCloud for oneAPI Documentation.

>**Note**: Since Intel® DevCloud for oneAPI includes the appropriate development environment already configured, you do not need to set environment variables.


## Guided Debugging

These instructions assume you have installed the Intel® Distribution for GDB* and have a basic working knowledge of GDB.

To learn how setup and use Intel® Distribution for GDB*, see the *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html)*.

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see *[Using the oneAPI Level Zero Interface: A Brief Introduction to the Level Zero API](https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-level-zero-interface.html?wapkw=Level%20Zero#gs.dxm4t4)*.

### Getting the Tracing and Profiling Tool

At an important step in this tutorial, the instructions require a utility that was not installed with the Intel® oneAPI Base Toolkit (Base Kit). 

You must download the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) code from GitHub and build the utility. The build instructions are included in the readme in the GitHub repository.

To complete the steps in the following section, you must have already built the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace). Once you have built the utility, you can invoke it before your program (similar to GDB).

### Guided Instructions for Zero Buffer

In `a1_matrix_mul_zero_buff`, a zero-element buffer is passed to a SYCL submit `lambda` function. **This will cause the application to crash.**

1. Run the program without the debugger.
   ```
   ./a1_matrix_mul_zero_buff
   ```
   The program should crash almost immediately in an exception as shown below.
   ```
   terminate called after throwing an instance of 'sycl::_V1::invalid_object_error'
     what():  SYCL buffer size is zero. To create a device accessor, SYCL buffer size must be greater than zero. -30 (PI_ERROR_INVALID_VALUE)
   Aborted (core dumped)

2. Start the debugger to watch the application failure and find out where it failed.
   ```
   $ gdb-oneapi ./a1_matrix_mul_zero_buff
   ```

3. You should get the prompt `(gdb)`.

4. From the debugger, run the program. The program will fail.
    ```
    (gdb) run
    ```
    Notice that you will see the same message when we ran it outside of the debugger.

    ```
    terminate called after throwing an instance of 'sycl::_V1::invalid_object_error'
      what():  SYCL buffer size is zero. To create a device accessor, SYCL buffer size must be greater than zero. -30 (PI_ERROR_INVALID_VALUE)
    Aborted (core dumped)
    ```

5. Run a `backtrace` to get a summary showing the approximate location triggering the assert.
   ```
   (gdb) backtrace
   ```

   The output can be extensive, and the output might look similar to the following.
   ```
   #0  0x00007ffff77c818b in raise () from /lib/x86_64-linux-gnu/libc.so.6
   #1  0x00007ffff77a7859 in abort () from /lib/x86_64-linux-gnu/libc.so.6
   #2  0x00007ffff7b84951 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #3  0x00007ffff7b9047c in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #4  0x00007ffff7b904e7 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #5  0x00007ffff7b90799 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x0000000000414204 in sycl::_V1::accessor<float, 2, (sycl::_V1::access::mode)1025, (sycl::_V1::access::target)2014, (sycl::_V1::access::placeholder)0, sycl::_V1::ext::oneapi::accessor_property_list<> >::preScreenAccessor(unsigned long, sycl::_V1::ext::oneapi::accessor_property_list<> const&) (this=0x7fffffffc138, elemInBuffer=0, PropertyList=...) at /home/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/../include/sycl/accessor.hpp:2165
   #7  0x0000000000413dd2 in sycl::_V1::accessor<float, 2, (sycl::_V1::access::mode)1025, (sycl::_V1::access::target)2014, (sycl::_V1::access::placeholder)0, sycl::_V1::ext::oneapi::accessor_property_list<> >::accessor<float, 2, sycl::_V1::detail::aligned_allocator<float>, void>(sycl::_V1::buffer<float, 2, sycl::_V1::detail::aligned_allocator<float>, std::enable_if<(((2)>(0)))&&((2)<=(3)), void>::type>&, sycl::_V1::handler&, sycl::_V1::property_list const&, sycl::_V1::detail::code_location) (this=0x7fffffffc138, Python Exception <class 'TypeError'>: expected string or bytes-like object, got 'NoneType' BufferRef=, CommandGroupHandler=..., PropertyList=..., CodeLoc=...) at /home/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/../include/sycl/accessor.hpp:1496
   #8  0x0000000000413b59 in sycl::_V1::accessor<float, 2, (sycl::_V1::access::mode)1025, (sycl::_V1::access::target)2014, (sycl::_V1::access::placeholder)0, sycl::_V1::ext::oneapi::accessor_property_list<> >::accessor<float, 2, sycl::_V1::detail::aligned_allocator<float>, sycl::_V1::mode_tag_t<(sycl::_V1::access::mode)1025>, void>(sycl::_V1::buffer<float, 2, sycl::_V1::detail::aligned_allocator<float>, std::enable_if<(((2)>(0)))&&((2)< (3)), void>::type>&, sycl::_V1::handler&, sycl::_V1::mode_tag_t<(sycl::_V1::access::mode)1025>, sycl::_V1::property_list const&, sycl::_V1::detail::code_location (this=0x7fffffffc138, Python Exception <class 'TypeError'>: expected string or bytes-like object, got 'NoneTy BadBuffers pe' BufferRef=, CommandGroupHandler=..., PropertyList=..., CodeLoc=...) at /home/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/../include/sycl/accessor.hpp:1550
   #9  0x0000000000408ff0 in main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const (this=0x7fffffffc510, h=... at /home/guided_matrix_mult_BadBuffers/src/a1_matrix_mul_zero_buff.cpp:76
   #10 0x0000000000408e55 in std::_Function_handler<void (sycl::_V1::handler&), main::{lambda(auto:1&)#1}>::_M_invoke(std::_Any_data const&, sycl::_V1::handler&) (__functor=..., __args=...) at /usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/bits/std_function.h:300
   #11 0x00007ffff7fb99b6 in sycl::_V1::detail::queue_impl::submit_impl(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/libsycl.so.6
   #12 0x00007ffff7fb8ff6 in sycl::_V1::detail::queue_impl::submit(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/libsycl.so.6
   #13 0x00007ffff7fb8fb5 in sycl::_V1::queue::submit_impl(std::function<void (sycl::_V1::handler&)>, sycl::_V1::detail::code_location const&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/libsycl.so.6
   #14 0x000000000040839f in sycl::_V1::queue::submit<main::{lambda(auto:1&)#1}>(main::{lambda(auto:1&)#1}, sycl::_V1::detail::code_location const&) (this=0x7fffffffc8d8, CGF=..., CodeLoc=...) at /home/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/../include/sycl/queue.hpp:326
   #15 0x0000000000408089 in main () at /home/guided_matrix_mult_BadBuffers/src/a1_matrix_mul_zero_buff.cpp:74
   ```
   
    Looking at the backtrace output, notice that the exception was triggered around line 74 of `a1_matrix_mul_zero_buff` (frame 15 in the example output listed shown above).

6. Examine the final frame (the frame number might be different from the output shown).
   ```
   (gdb) frame 15
   ```

7. Examine the code in that region.
   ```
   (gdb) list
   ```
   The code in Frame 16 that triggered the assert is the `submit` lambda at line 74:
   ```
   73          // Submit command group to queue to initialize matrix a
   74          q.submit([&](auto &h) {
   75            // Get write only access to the buffer on a device.
   76            accessor a(a_buf, h, write_only);
   77
   78            // Execute kernel.
   79            h.parallel_for(range(M, N), [=](auto index) {
   80              // Each element of matrix a is 1.
   81              a[index] = 1.0f;
   82            });
   83          });
   ```

   This isn't very helpful since anything in the `submit` might have been the cause. Looking a little deeper into the backtrace, we see that the exception happened in the accessor code (frame 6), so that indicates something was wrong in line 76 (the only accessor code in this `submit).

8. Inspect `a_buf` in the debugger using the `print` command. 
   ```
   (gdb) print /r a_buf
   ```
   >**Note:** `/r` disables the *pretty printer* for the SYCL `buffer` class. You can see all available pretty printers using `info pretty-printer` at the gdb prompt.

   You might notice that this buffer has a size 0 by 0 elements; see the `Range` field. Since it has zero size, this buffer is the problem.

   ```
   $1 = {impl = warning: RTTI symbol not found for class 'std::_Sp_counted_ptr_inplace<cl::sycl::detail::buffer_impl, std::allocator<cl::sycl::detail::buffer_impl>, (__gnu_cxx::_Lock_policy)2>'
   warning: RTTI symbol not found for class 'std::_Sp_counted_ptr_inplace<cl::sycl::detail::buffer_impl, std::allocator<cl::sycl::detail::buffer_impl>, (__gnu_cxx::_Lock_policy)2>'
   std::shared_ptr<cl::sycl::detail::buffer_impl> (use count 1, weak count 0) = {get() = 0xed9160}, Range = {<cl::sycl::detail::array<2>> = {common_array = {0, 
           0}}, <No data fields>}, OffsetInBytes = 0, IsSubBuffer = false}
   ```

9. Compare the `b_buf` to the previous buffer output.
   ```
   (gdb) print /r b_buf
   ```
   The output should display similar to the following.
   ```
   $2 = {impl = warning: RTTI symbol not found for class 'std::_Sp_counted_ptr_inplace<cl::sycl::detail::buffer_impl, std::allocator<cl::sycl::detail::buffer_impl>, (__gnu_cxx::_Lock_policy)2>'
   warning: RTTI symbol not found for class 'std::_Sp_counted_ptr_inplace<cl::sycl::detail::buffer_impl, std::allocator<cl::sycl::detail::buffer_impl>, (__gnu_cxx::_Lock_policy)2>'
   std::shared_ptr<cl::sycl::detail::buffer_impl> (use count 1, weak count 0) = {get() = 0xed9230}, Range = {<cl::sycl::detail::array<2>> = {common_array = {300, 
           600}}, <No data fields>}, OffsetInBytes = 0, IsSubBuffer = false}
   ```

   You will see this variable has a `Range` field of 300 by 600 elements, so it is an array with a non-zero size.

   Looking at the source again, you'll see that this originated when the buffers were created around line 61 and 62:
   ```
    buffer<float, 2> a_buf(range(0, 0));
    buffer<float, 2> b_buf(range(N, P));
   ```

   In real code the values to the ranges may be passed into the function from outside, so you will need to inspect those as well as the code where they are calculated.  For example, you would need to find the values of `M`, `N`, and `P` to make sure that the resulting buffer sizes are non-zero:
   ```
    buffer<float, 2> a_buf(range(M, N));
    buffer<float, 2> b_buf(range(N, P));
   ```

### Guided Instructions for Null Device Pointer

In `b1_matrix_mul_null_usm.cpp` a bad (in this case, null) pointer that is supposed to represent memory allocated on the device is inadvertently used in a kernel. This bug is a little harder to root-cause than the one we just saw.

You will run the application and review what happens when the program is run using each of the runtimes.

1. Run the program on the GPU using Level Zero.
   ```
   SYCL_DEVICE_FILTER=level_zero:gpu ./b1_matrix_mul_null_usm
   ```
   This run produces troublesome output.
   ```
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Result of matrix multiplication using DPC++:
   Fail - The result is incorrect for element: [0, 0], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 1], expected: 45150, but found: :
   Fail - The results mismatch!
   ```

2. Check the output if we run on the GPU again but using OpenCL.
   ```
   SYCL_DEVICE_FILTER=opencl:gpu ./b1_matrix_mul_null_usm
   ```

   The results should be the same as the Level Zero output.

3. Check the output we get by bypassing the GPU entirely and using the OpenCL driver for CPU.
   ```
   SYCL_DEVICE_FILTER=opencl:cpu ./b1_matrix_mul_null_usm
   ```
   ```
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Segmentation fault (core dumped)
   ```
   That's even worse than getting mismatched results!

#### Attempting to Understand What Is Happening

Looking at those incorrect results from the GPU, your first thought might have been that we have a race condition.  But if that were the case, the CPU should have exited with bad results as well, rather than giving a segmentation fault.

Let's see what caused the problem by running in the debugger using the OpenCL driver for the CPU.

1. Start the debugger using OpenCL™ on the CPU.
   ```
   SYCL_DEVICE_FILTER=opencl:cpu gdb-oneapi ./b1_matrix_mul_null_usm
   ```
2. You should get the prompt `(gdb)`.

3. From the debugger, run the program.
   ```
   (gdb) run
   ```
   This will launch the application, and quickly generate an error.

   ```
   Problem size: c(150,600) = a(150,300) * b(300,600)

   Thread 1 "b1_matrix_mul_n" received signal SIGSEGV, Segmentation fault.
   0x00007fffe3e14262 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> > (sycl::_V1::item<2, true>) const (this=0x7fffffffa738, index=...) at b1_matrix_mul_null_usm.cpp:122
   122               sum += dev_a[a_index] * dev_b[b_index];
   (gdb)
   ```

4. Run the backtrace to see where we run into a problem.
   ```
   (gdb) backtrace
   ```
   You should see output similar to the following.
   ```
   #0  0x00007fffe3e14262 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> >(sycl::_V1::item<2, true>) const (this=0x7fffffffa738, index=...) at b1_matrix_mul_null_usm.cpp:122
   #1  0x00007fffe3e14c8c in _ZTSZZ4mainENKUlRT_E0_clIN4sycl3_V17handlerEEEDaS0_EUlS_E_ (_arg_width_a=300, _arg_dev_a=0x0, _arg_dev_b=0x7fffd8a25000, _arg_dev_c=0x7fffd89cd000) at /home/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/../include/sycl/handler.hpp:1202
   #2  0x00007fffe1ab9c95 in Intel::OpenCL::DeviceBackend::Kernel::RunGroup(void const*, unsigned long const*, void*) const () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #3  0x00007fffe1aba2ea in non-virtual thunk to Intel::OpenCL::DeviceBackend::Kernel::RunGroup(void const*, unsigned long const*, void*) const () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #4  0x00007fffe1c142f7 in Intel::OpenCL::CPUDevice::NDRange::ExecuteIteration(unsigned long, unsigned long, unsigned long, void*) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #5  0x00007fffdd273d99 in tbb::detail::d1::start_for<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>, TaskLoopBody2D<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit> >, tbb::detail::d1::auto_partitioner const>::run_body(Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libtask_executor.so.2023.15.2.0
   #6  0x00007fffdd273a5a in void tbb::detail::d1::dynamic_grainsize_mode<tbb::detail::d1::adaptive_mode<tbb::detail::d1::auto_partition_type> >::work_balance<tbb::detail::d1::start_for<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>, TaskLoopBody2D<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit> >, tbb::detail::d1::auto_partitioner const>, Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit> >(tbb::detail::d1::start_for<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>, TaskLoopBody2D<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit> >, tbb::detail::d1::auto_partitioner const>&, Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>&, tbb::detail::d1::execution_data&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libtask_executor.so.2023.15.2.0
   #7  0x00007fffdd27363e in tbb::detail::d1::start_for<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>, TaskLoopBody2D<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit> >, tbb::detail::d1::auto_partitioner const>::execute(tbb::detail::d1::execution_data&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libtask_executor.so.2023.15.2.0
   #8  0x00007fffe3b88ab0 in tbb::detail::r1::task_dispatcher::local_wait_for_all<false, tbb::detail::r1::external_waiter> (this=0x7fffdd061c00, t=0x7fffdd03f700, waiter=...) at /localdisk/ci/runner008/intel-innersource/001/_work/libraries.threading.infrastructure.onetbb-ci/libraries.threading.infrastructure.onetbb-ci/onetbb t/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libtask_executor.so.2023.15.2.0
   ...
   #19 0x00007fffdd26b460 in Intel::OpenCL::TaskExecutor::out_of_order_command_list::LaunchExecutorTask(bool, Intel::OpenCL::Utils::SharedPtr<Intel::OpenCL::TaskExecutor::ITaskBase> const&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libtask_executor.so.2023.15.2.0
   #20 0x00007fffdd26acac in Intel::OpenCL::TaskExecutor::base_command_list::InternalFlush(bool) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libtask_executor.so.2023.15.2.0
   #21 0x00007fffdd26ab1d in Intel::OpenCL::TaskExecutor::base_command_list::WaitForCompletion(Intel::OpenCL::Utils::SharedPtr<Intel::OpenCL::TaskExecutor::ITaskBase> const&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libtask_executor.so.2023.15.2.0
   #22 0x00007fffe03787e7 in Intel::OpenCL::CPUDevice::CPUDevice::clDevCommandListWaitCompletion(void*, cl_dev_cmd_desc*) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #23 0x00007fffdfcbd0f9 in Intel::OpenCL::Framework::IOclCommandQueueBase::WaitForCompletion(Intel::OpenCL::Utils::SharedPtr<Intel::OpenCL::Framework::QueueEvent> const&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #24 0x00007fffdfca3bb6 in Intel::OpenCL::Framework::ExecutionModule::Finish(Intel::OpenCL::Utils::SharedPtr<Intel::OpenCL::Framework::IOclCommandQueueBase> const&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #25 0x00007fffdfca3946 in Intel::OpenCL::Framework::ExecutionModule::Finish(_cl_command_queue*) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #26 0x00007fffdf907c7d in clFinish () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/x64/libintelocl.so
   #27 0x00007ffff7eb3f72 in _pi_result sycl::_V1::detail::plugin::call_nocheck<(sycl::_V1::detail::PiApiKind)23, _pi_queue*>(_pi_queue*) const () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/libsycl.so.6
   #28 0x00007ffff7f1465a in sycl::_V1::detail::queue_impl::wait(sycl::_V1::detail::code_location const&) () from /home/opt/intel/oneapi/compiler/2023.1.0/linux/lib/libsycl.so.6
   #29 0x000000000040b8fd in sycl::_V1::queue::wait (this=0x7fffffffc880, CodeLoc=...) at /home/gta/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/../include/sycl/queue.hpp:435
   #30 0x00000000004084e8 in main () at /home/guided_matrix_mult_BadBuffers/src/b1_matrix_mul_null_usm.cpp:130
   (gdb)
   ```

5. The output is not helpful, so let us drill down into the last frame.
   ```
   (gdb) frame 30
   ```
   Output should be similar to the following.
   ```
   #30 0x00000000004084e8 in main () at /home/guided_matrix_mult_BadBuffers/src/b1_matrix_mul_null_usm.cpp:130
   130         q.wait();
   ```

6. Examine the code in that region.
   ```
   (gdb) list
   ```
   You should see the code around the line reporting the problem.

   ```
   125             auto idx = row * P + col;
   126             dev_c[idx] = sum;
   127           });
   128         });
   129
   130         q.wait();
   131
   132         q.memcpy(&c_back[0], dev_c, M*P * sizeof(float));
   133
   134         q.wait();
   ```
   This situation is a little confusing. Frame 30 in the backtrace would suggest that we crash while waiting for the kernel to complete. We know that the kernels operate synchronously (hence the `q.wait`), but this doesn't help us root-cause the issue at all.  We need to look deeper into the stack to see what is actually going on.

7. Look at Frame 0, which is at line 122 in our code, and not somewhere deep in the SYCL runtime!
   ```
   (gdb) frame 0
   ```
   The output should look similar to the following.
   ```
   #0  0x00007fffe3e14262 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> >(sycl::_V1::item<2, true>) const (this=0x7fffffffa738, index=...) at b1_matrix_mul_null_usm.cpp:122
   122               sum += dev_a[a_index] * dev_b[b_index];

   ```

8. Examine the source code in that region.
   ```
   (gdb) list
   ```
   You should see the code around line 122.
   ```
   117
   118             // Compute the result of one element of c
   119             for (int i = 0; i < width_a; i++) {
   120               auto a_index = row * width_a + i;
   121               auto b_index = i * P + col;
   122               sum += dev_a[a_index] * dev_b[b_index];
   123             }
   124
   125             auto idx = row * P + col;
   126             dev_c[idx] = sum;
   ```

9. Let's check the pointers in use at line 122.
   1. Check the first variable. 
      ```
      (gdb) print dev_a
      ```
      This should display output similar to the following.
      ```
      $1 = (float *) 0x0
      ```
   2. Check the second variable.
      ```
      (gdb) print dev_b
      ```
      This should display output similar to the following.
      ```
      $2 = (float *)  0x7fffd8a25000
      ```
   This located the problem since one of the array pointers is null (zero) while the other has a (hopefully) valid value. If both pointers were valid, we'd want to check the index values being used to access the arrays, and the memory values at those locations (and at the start of the array).

#### Understanding What Is Happening

Long ago, operating system designers realized that it was common for developers to write bugs in which they accidentally try to de-reference null pointers (access them like an array).  To make it easier to find these sorts of errors the operating system designers implemented logic that made it illegal for any program to access the first two memory pages (so from address 0x0 to around 0x2000).  They didn't go further to explicitly validate the address of any memory accessed by a program (because it is too expensive), but this range of illegal addresses was a cheap check that caught a huge number of bugs.

GPUs typically don't have a lot of memory, so they can't afford to set aside a large range of illegal addresses.  So both Level Zero and OpenCL passed on the null pointer to device memory assuming it was valid, and it would have been valid if that memory had been actually allocated on the device at address zero.   But it was not, so we accessed random memory values in calculating the sum on line 122, and returned incorrect results as a consequence.   Only running on the CPU caught this mistake.

What would we have seen if `dev_a` had contained just a random pointer value?  On the CPU, if you were lucky, the address returned when you print `dev_a` would look very different from the one returned when you printed `dev_b`, but that's not a certainty.  Or it would have pointed to memory not owned by your process and caused a crash.  On the GPU side, you will get something different:
```
terminate called after throwing an instance of 'cl::sycl::runtime_error'
  what():  Native API failed. Native API returns: -50 (CL_INVALID_ARG_VALUE) -50 (CL_INVALID_ARG_VALUE)
```
In the call stack you will see the crash at the submit statement with the bad variable, but no information on the variable that caused the error.

#### Other Debug Techniques

If you are trying to debug bad values on the GPU, you will be very challenged to figure out whether or not this is due to bad inputs. You *might* be able to catch the bad pointer if you breakpoint at the correct line. You also *might* see something odd if you capture the API calls with `onetrace`. 

You need to build onetrace before you can use it. See the instructions at [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace). Once you have built the utility, you can invoke it before your program (similar to GDB).

1. Run the oneTrace tool. (Include `-c` when invoking `onetrace` to enable call logging of API calls.)

   >**Note**: You must modify the command shown below to include the path to where you installed the `onetrace` utility.
   ```
   [path]/onetrace -c ./b1_matrix_mul_null_usm > b1_matrix_mul_null_usm_onetrace_out.txt 2>&1
   ```
   While reviewing the output, you might see something like the following excerpt in the output.
   ```
   :
   >>>> [198147949] zeKernelCreate: hModule = 0x2b127b0 desc = 0x7fffd0375630 {ZE_STRUCTURE_TYPE_KERNEL_DESC(0x1d) 0 0 "_ZTSZZ4mainENKUlRT_E0_clIN2cl4sycl7handlerEEEDaS0_EUlS_E_"} phKerne
   l = 0x7fffd0375628 (hKernel = 0x2af4720)
   <<<< [198599544] zeKernelCreate [448864 ns] hKernel = 0x2c487a0 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [198608874] zeKernelSetIndirectAccess: hKernel = 0x2c487a0 flags = 7
   <<<< [198611092] zeKernelSetIndirectAccess [735 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [198619429] zeKernelSetArgumentValue: hKernel = 0x2c487a0 argIndex = 0 argSize = 4 pArgValue = 0x2b11870
   <<<< [198621003] zeKernelSetArgumentValue [305 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [198625048] zeKernelSetArgumentValue: hKernel = 0x2c487a0 argIndex = 1 argSize = 8 pArgValue = 0
   <<<< [198627236] zeKernelSetArgumentValue [1072 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [198630079] zeKernelSetArgumentValue: hKernel = 0x2c487a0 argIndex = 2 argSize = 8 pArgValue = 0x2b11880
   <<<< [198635408] zeKernelSetArgumentValue [4305 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [198637750] zeKernelSetArgumentValue: hKernel = 0x2c487a0 argIndex = 3 argSize = 8 pArgValue = 0x2b11888
   <<<< [198639962] zeKernelSetArgumentValue [1207 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [198645602] zeKernelGetProperties: hKernel = 0x2c487a0 pKernelProperties = 0x34b17c0
   <<<< [198647342] zeKernelGetProperties [693 ns] -> ZE_RESULT_SUCCESS(0x0)
   :
   ```
   Notice how the kernel arguments have a non-zero value except one (`argIndex = 1`) when they are set up. We have nothing to map the kernel arguments created by the SYCL runtime and passed to Level Zero to the arguments of the user program (the value may be OK), but it's suspicious. Likewise, if you see a kernel argument that looks out of place (note that the values at all other argument indexes are very similar), you might want to be suspicious.

   Other than this clue, there are no other hints that the bad output from the program is due to a bad input argument rather than a race condition or other algorithmic error.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
