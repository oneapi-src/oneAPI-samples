# `Guided Matrix Multiplication Illegal SLM Size` Sample

The `Guided Matrix Multiplication Illegal SLM Size` sample demonstrates a guided approach to debugging incorrect use of the SYCL language. The sample uses the Intel® oneAPI Base Toolkit (Base Kit) and several tools included in the Base Kit.

The sample is a simple program that multiplies together two large matrices and verifies the results.

| Area                  | Description
|:---                   |:---
| What you will learn   | A method to root-cause incorrect use of queues with different contexts.
| Time to complete      | 50 minutes
| Category              | Tutorial

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The sample in this tutorial shows how to debug crashes that occur when the user tries to reserve more memory for a work-group than there is space in work-group local memory (also called [Shared Local Memory (SLM)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/kernels/slm.html)).

Using this type of memory when working with GPUs is an important optimization, but you must be careful due to its limited size. Shared local memory is often also shared with/traded for Vector Engine memory registers.

The sample includes different versions of a simple matrix multiplication program.

| File name                    |  Description
|:---                          |:---
| `1_matrix_mul_SLM_size.cpp`  | This example shows an extremely artificial example of this problem.
| `2_matrix_mul.cpp`           | A working version of the matrix multiply code where all work-group local memory operations fit within the SLM.

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                  | Ubuntu* 20.04
| Hardware            | GEN9 or newer
| Software            | Intel® oneAPI DPC++/C++ Compiler <br> Intel® Distribution for GDB* <br> [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace), which is available from the [onetrace](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) GitHub repository.

## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* queues and devices
- Allocation of work-group local memory via a local_accessor class
- SYCL* kernels (including parallel_for function and explicit memory copies)
- SYCL* queues

The type of error shown in this sample can be hard to detect and root cause in a large body of code where large amounts of data are passed due to the lack of tools that tell you what is actually going wrong, and because the resulting error message ("`PI_ERROR_OUT_OF_RESOURCES`") isn't informative.

This can be particularly painful. For example, you might experience this error in the cases where code that runs on a device with a large amount of shared local memory fails on a device with less shared local memory, or where one data set out of many results in an allocation that exceeds SLM limits on a given machine.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build and Run the `Guided Matrix Multiplication Illegal SLM Size` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.


### On Linux*

1. Change to the sample directory.
2. Build the programs.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the program.
   ```
   make run_all
   ```
   > **Note**: The application by default uses the Level Zero runtime and will run without errors.  We will do a deeper investigation of the application, in particular with the openCL runtime, to expose problems that could also manifest in Level Zero.

   For the broken SLM version only, enter the following:
   ```
   make run_bugged
   ```
   For the working version only, enter the following:
   ```
   make run_OK
   ```
4. Clean the program. (Optional)
   ```
   make clean
   ```

### Troubleshooting

If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

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

The following instructions assume you have installed Intel® Distribution for GDB* and have a basic working knowledge of GDB.

To learn how setup and use Intel® Distribution for GDB*, see *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html)*.

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see *[Using the oneAPI Level Zero Interface: A Brief Introduction to the Level Zero API](https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-level-zero-interface.html)*.


### Getting the Tracing and Profiling Tool

At an important step in this tutorial, the instructions require a utility that was not installed with the Intel® oneAPI Base Toolkit (Base Kit).

You must download the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) code from GitHub and build the utility. The build instructions are included in the readme in the GitHub repository.

### Check the Program

In `1_matrix_mul_SLM_size`, the local_accessor class is used to reserve an illegal amount of device-local memory. If you attempt to run the code, the application will crash.

#### Observe the Failure

1. Run the program outside the debugger.
   ```
   ./1_matrix_mul_SLM_size
   ```

2. It should almost immediately crash in an exception:
   ```
   $ ./1_matrix_mul_SLM_size
   Initializing
   Computing

   Problem size: c(150,600) = a(150,300) * b(300,600)
   terminate called after throwing an instance of 'sycl::_V1::runtime_error'
     what():  Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)
   Aborted (core dumped)
   ```

#### Locate the General Location of the Problem

1. Start the debugger to learn more about the error.
   ```
   gdb-oneapi ./1_matrix_mul_SLM_size
   ```

2. Run the application within the debugger.
   ```
   (gdb) run
   ```
   The application will fail and display the same message when we ran it outside of the debugger.

   ```
   Problem size: c(150,600) = a(150,300) * b(300,600)
   terminate called after throwing an instance of 'sycl::_V1::runtime_error'
     what():  Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)

   Thread 1 "1_matrix_mul_SL" received signal SIGABRT, Aborted.
   0x00007ffff779b18b in raise () from /lib/x86_64-linux-gnu/libc.so.6
   (gdb)
   ```

3. Run `backtrace` to get a summary showing the rough location that triggered the error.
   ```
   (gdb) backtrace
   ```

   Looking at the backtrace output, we can see that the error happened around line 104 (frame 15):
   ```
   (gdb) backtrace
   #0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
   #1  0x00007ffff76ea859 in __GI_abort () at abort.c:79
   #2  0x00007ffff7aca8d1 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #3  0x00007ffff7ad637c in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #4  0x00007ffff7ad63e7 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #5  0x00007ffff7ad6699 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x00007ffff7db9d77 in sycl::_V1::detail::enqueue_kernel_launch::handleErrorOrWarning(_pi_result, sycl::_V1::detail::device_impl const&, _pi_kernel*, sycl::_V1::detail::NDRDescT const&) () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #7  0x00007ffff7e9ac74 in sycl::_V1::detail::enqueueImpKernel(std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::NDRDescT&, std::vector<sycl::_V1::detail::ArgDesc, std::allocator<sycl::_V1::detail::ArgDesc> >&, std::shared_ptr<sycl::_V1::detail::kernel_bundle_impl> const&, std::shared_ptr<sycl::_V1::detail::kernel_impl> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::vector<_pi_event*, std::allocator<_pi_event*> >&, std::shared_ptr<sycl::_V1::detail::event_impl> const&, std::function<void* (sycl::_V1::detail::AccessorImplHost*)> const&, _pi_kernel_cache_config) () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #8  0x00007ffff7eed66d in sycl::_V1::handler::finalize()::$_0::operator()() const () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #9  0x00007ffff7eeaa22 in sycl::_V1::handler::finalize() () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #10 0x00007ffff7e6a51b in void sycl::_V1::detail::queue_impl::finalizeHandler<sycl::_V1::handler>(sycl::_V1::handler&, sycl::_V1::event&) () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #11 0x00007ffff7e6a0d9 in sycl::_V1::detail::queue_impl::submit_impl(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #12 0x00007ffff7e69ac0 in sycl::_V1::detail::queue_impl::submit(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #13 0x00007ffff7f1c7f5 in sycl::_V1::queue::submit_impl(std::function<void (sycl::_V1::handler&)>, sycl::_V1::detail::code_location const&) () from /opt/intel/oneapi/compiler/2024.2/lib/libsycl.so.7
   #14 0x00000000004046ba in sycl::_V1::queue::submit<main::{lambda(sycl::_V1::handler&)#1}>(main::{lambda(sycl::_V1::handler&)#1}, sycl::_V1::detail::code_location const&) (this=0x7fffffffd1f0, CGF=..., CodeLoc=...) at /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/queue.hpp:366
   #15 0x00000000004041b8 in main () at 1_matrix_mul_SLM_size.cpp:104
   ```

4. Look at the final frame. (Your frame number might differ).
   ```
   (gdb) frame 15
   ```

5. Examine the code in that region of code that caused the crash.
   ```
   (gdb) list
   ```
   The code that triggered the error is found at line `104` (in the example output below):  a `submit` lambda.

   ```
   99              h.memcpy(dev_c, &c_back[0], M*P * sizeof(float));
   100         });
   101
   102         q.wait();
   103
   104         q.submit([&](handler &h){
   105           local_accessor<float,1> acc(163850, h);
   106           h.parallel_for(nd_range<1>{{163850}, {10}}, [=](nd_item<1> i){
   107               int index = i.get_global_id();
   108               acc[index] = index;
   (gdb)
   ```

#### Root-Cause the Issue

You can see that there is something wrong in the submit at line `104`, but we need more information to understand what is happening. For that we need to capture the lower-level API calls using the `onetrace` tool.

>**Note**: You must have already built the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace). Once you have built the utility, you can invoke it before your program (similar to GBD).

Among other things, the Tracing and Profiling utility can print every low-level API call made to OpenCL™ or Level Zero. This is the feature that we will use to get more information about the crash.

2. Run the program with `onetrace` and enable the runtime debug messages:
   ```
   onetrace -c ./1_matrix_mul_SLM_size
   ```

3. Let the output continue until the error occurs and the program stops.
   ```
   :
   >>>> [1066578697845396] zeKernelSetGroupSize: hKernel = 55242736 groupSizeX = 10 groupSizeY = 1 groupSizeZ = 1
   <<<< [1066578697849285] zeKernelSetGroupSize [1449 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1066578697854047] zeCommandListCreateImmediate: hContext = 41540224 hDevice = 37134192 altdesc = 140733241819552 {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC(0xe) 0 0 0 0 2 0} phCommandList = 140733241819544 (hCommandList = 0)
   <<<< [1066578698107437] zeCommandListCreateImmediate [248694 ns] hCommandList = 61984688 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1066578698115446] zeEventHostReset: hEvent = 39536208
   <<<< [1066578698119590] zeEventHostReset [1854 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1066578698126085] zeCommandListAppendLaunchKernel: hCommandList = 61984688 hKernel = 55242736 (_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_) pLaunchFuncArgs = 140733241820008 {16385, 1, 1} hSignalEvent = 39536208 numWaitEvents = 0 phWaitEvents = 0
   <<<< [1066578698169233] zeCommandListAppendLaunchKernel [34637 ns] -> ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY(0x1879048195)
   terminate called after throwing an instance of 'sycl::_V1::runtime_error'
     what():  Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)
   Aborted (core dumped)
   ```

  **Clue**: By running the program under onetrace we can see that the error happens when launching a kernel called `(_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_`), and that this fails with an `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` error.

   A note about the output above. You will see that is has two lines that read:

   ```
   >>>> [1066578697845396] zeKernelSetGroupSize: hKernel = 55242736 groupSizeX = 10 groupSizeY = 1 groupSizeZ = 1
   :
   >>>> [1066578698126085] zeCommandListAppendLaunchKernel: hCommandList = 61984688 hKernel = 55242736 (_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_) pLaunchFuncArgs = 140733241820008 {16385, 1, 1} hSignalEvent = 39536208 numWaitEvents = 0 phWaitEvents = 0
   ```

   We used the form of `parallel_for` that takes the `nd_range`, which specifies the global iteration range (163850) and the local work-group size (10) like so:  `nd_range<1>{{163850}, {10}}`. The first line above shows the workgroup size (`groupSizeX = 10 groupSizeY = 1 groupSizeZ = 1`), and the second shows how many total workgroups will be needed to process the global iteration range (`{16385, 1, 1}`).

#### Determine Device Limits

If you have access to a version of the graphics drivers built with debug functionality, you can get even more information about this error by setting two NEO variables to the following values:

```
export NEOReadDebugKeys=1
export PrintDebugMessages=1
```

When you set these environment variables and re-run the program, you should see results similar to the following:

```
./1_matrix_mul_SLM_size
Initializing
:Problem size: c(150,600) = a(150,300) * b(300,600)
Ignored kernel-scope Patch Token: 21
Ignored kernel-scope Patch Token: 21
Ignored kernel-scope Patch Token: 21
Ignored kernel-scope Patch Token: 21
Size of SLM (656384) larger than available (131072)
terminate called after throwing an instance of 'sycl::_V1::runtime_error'
  what():  Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)
Aborted (core dumped)
```

The new message of interest is `Size of SLM (656384) larger than available (131072)`. This tells you that the size of the Shared Local Memory (SLM) memory on the device, 131072 bytes (128Kb), is smaller than the requested size of 656384 bytes.

If the `parallel_for` were operating over a multi-dimensional range (for example, if `acc` were two or three-dimensional), you need to multiply the dimensions together to determine the number of floating point numbers we are trying to store in SLM. In our case, the calculation is easy:  the first argument to the `nd_range` in the `parallel_for` is single-dimensional, so it's just 163850. Thus the problem is that the size of work-group local memory we tried to allocate, (163850 floats or 4*163850=655,400 bytes rounded up to the nearest 64-byte cache line), doesn't fit in the SLM on this device.

You should know that different devices will have different amounts of memory set aside as SLM. In SYCL, you can query this number by passing `info::device::local_mem_size` to the `get_info` member of the `device` class.

Finally, running under `onetrace -c` you see:

```
>>>> [1066578697845396] zeKernelSetGroupSize: hKernel = 55242736 groupSizeX = 10 groupSizeY = 1 groupSizeZ = 1
<<<< [1066578697849285] zeKernelSetGroupSize [1449 ns] -> ZE_RESULT_SUCCESS(0x0)
>>>> [1066578697854047] zeCommandListCreateImmediate: hContext = 41540224 hDevice = 37134192 altdesc = 140733241819552 {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC(0xe) 0 0 0 0 2 0} phCommandList = 140733241819544 (hCommandList = 0)
<<<< [1066578698107437] zeCommandListCreateImmediate [248694 ns] hCommandList = 61984688 -> ZE_RESULT_SUCCESS(0x0)
>>>> [1066578698115446] zeEventHostReset: hEvent = 39536208
<<<< [1066578698119590] zeEventHostReset [1854 ns] -> ZE_RESULT_SUCCESS(0x0)
>>>> [1066578698126085] zeCommandListAppendLaunchKernel: hCommandList = 61984688 hKernel = 55242736 (_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_) pLaunchFuncArgs = 140733241820008 {16385, 1, 1} hSignalEvent = 39536208 numWaitEvents = 0 phWaitEvents = 0
Size of SLM (656384) larger than available (131072)
<<<< [1066578698169233] zeCommandListAppendLaunchKernel [34637 ns] -> ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY(0x1879048195)
terminate called after throwing an instance of 'sycl::_V1::runtime_error'
  what():  Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)
```

This is useful because it shows you the kernel being called that caused the error (`_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_` which `c++filt` resolves to `typeinfo name for main::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1} `) in addition to the amount of memory requested vs. the available size of SLM.



#### Resolving the Problem

The synthetic code in this example has nothing to do with matrix multiply and can simply be removed to resolve the problem, so you can delete code to solve the problem.

1. Delete the code from line `104` to `110`.
   ```
   104    q.submit([&](handler &h){
   105     local_accessor<float,1> acc(163850, h);
   106     h.parallel_for(nd_range<1>{{163850}, {10}}, [=](nd_item<1> i){
   107         int index = i.get_global_id();
   108         acc[index] = index;
   109       });
   110   }).wait();
   ```

In real code, now that we have deduced the variable that is the source of the problem (the `acc` array in our synthetic code, which is defined as a `local_accessor`, meaning it will be stored in device shared-local memory), the "fix" is to rethink your algorithm. For example, can you break up `acc` into smaller sections that will fit on SLM and operate on them separately, one after the other?  You should determine whether `acc` really needs to be in work-group local memory.

As noted in *Shared Local Memory* topic of the *[oneAPI GPU Optimization Guide
Developer Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/kernels/slm.html)*, this really only makes sense when work-items need to share data and communicate with each other within a work-group.

In the synthetic code shown above, none of this is happening (each iteration is independent of every other since `i` is a SYCL `id` class, meaning that `i.get_id()` returns a unique index for each of the 163850 iterations). There is no reason why `acc` needs to be in work-group local memory. Instead, `acc` could be a normal `accessor` with a `device` target and a `read_write` access mode that would live in device global memory.

If data-sharing and communication between work-items is required, then you will need to find a way to break up the problem in such a way that the required SLM doesn't exceed device limits (for example, use a "window" or buffer into a larger array).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
