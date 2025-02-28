# `Guided Matrix Multiplication Illegal SLM Size` Sample

The `Guided Matrix Multiplication Illegal SLM Size` sample demonstrates a guided approach to debugging incorrect use of the SYCL language. The sample uses the Intel® oneAPI Base Toolkit (Base Kit) and several tools included in the Base Kit.

The sample is a simple program that multiplies together two large matrices and verifies the results.

| Area                  | Description
|:---                   |:---
| What you will learn   | A method to root-cause incorrect use of queues with different contexts.
| Time to complete      | 50 minutes
| Category              | Tutorial

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The sample in this tutorial shows how to debug crashes that occur when the user tries to reserve more memory for a work-group than there is space in work-group local memory (also called [Shared Local Memory (SLM)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/shared-local-memory.html).

Using this type of memory when working with GPUs is an important optimization, but you must be careful due to its limited size. Shared local memory is often also shared with/traded for Vector Engine memory registers.

The sample includes different versions of a simple matrix multiplication program.

| File name                    |  Description
|:---                          |:---
| `1_matrix_mul_SLM_size.cpp`  | This example shows an extremely artificial example of this problem.
| `2_matrix_mul.cpp`           | A working version of the matrix multiply code where all work-group local memory operations fit within the SLM.

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                      | Ubuntu* 24.04 LTS
| Hardware                | GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler 2025.0 <br> Intel® Distribution for GDB* 2025.0 <br> Unified Tracing and Profiling Tool 2.1.2, which is available from the [following Github repository](https://github.com/intel/pti-gpu/tree/master/tools/unitrace).
| Intel GPU Driver | Intel® General-Purpose GPU Rolling Release driver 2506.18 or newer from https://dgpu-docs.intel.com/releases/releases.html

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
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/use-the-setvars-and-oneapi-vars-scripts-with-linux.html)*.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension *Environment Configurator for Intel® Software Development Tools*.

 2. Download a sample using the extension *Code Sample Browser for Intel® Software Development Tools*.

 3. Open a terminal in VS Code (`Terminal > New Terminal`).

 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the *[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-vs-code/current/overview.html)*.

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

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html)* for more information on using the utility.

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

To learn how setup and use Intel® Distribution for GDB*, see the *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/get-started-guide-linux/current/overview.html)*.

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see the *[Level Zero Specification Documentation - Introduction](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html)* and *[Intel® oneAPI Level Zero](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/intel-oneapi-level-zero.html)*.


### Getting the Tracing and Profiling Tool

At a step in this tutorial, the instructions require a utility that was not installed with the Intel® oneAPI Base Toolkit (Base Kit). 

To complete the steps in the following section, you must download the [Unified Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) code from GitHub and build the utility. The build instructions are included in the README in the GitHub repository.  This build will go much more smoothly if you first install the latest drivers from [the Intel GPU driver download site](https://dgpu-docs.intel.com/driver/overview.html), especially the development packages (only available in the Data Center GPU driver install ).  Once you have built the utility, you invoke it on the command line in front of your program (similar to using GDB).

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
   Device: Intel(R) Data Center GPU Max 1550
   Device compute units: 512
   Device max work item size: 1024, 1024, 1024
   Device max work group size: 1024
   Problem size: c(150,600) = a(150,300) * b(300,600)
   terminate called after throwing an instance of 'sycl::_V1::exception'
     what():  UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
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
   The application will fail and display the same message when we ran it outside of the debugger (Please ignore the `Debugging of GPU offloaded code is not enabled.` error messages and answer "n" when gdb-oneapi asks `Quit anyway? (y or n)`).

   ```
   :
   Problem size: c(150,600) = a(150,300) * b(300,600)
   terminate called after throwing an instance of 'sycl::_V1::exception'
     what():  UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)

   Thread 1.1 "1_matrix_mul_SL" received signal SIGABRT, Aborted.
   (gdb)
   ```

3. Run `backtrace` to get a summary showing the rough location that triggered the error.
   ```
   (gdb) backtrace
   ```

   Looking at the backtrace output, we can see that the error happened around line 104 (frame 15):
   ```
   (gdb) backtrace
   #0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
   #1  __pthread_kill_internal (signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:78
   #2  __GI___pthread_kill (threadid=<optimized out>, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
   #3  0x00007ffff744527e in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
   #4  0x00007ffff74288ff in __GI_abort () at ./stdlib/abort.c:79
   #5  0x00007ffff78a5ff5 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x00007ffff78bb0da in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #7  0x00007ffff78a5a55 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #8  0x00007ffff78bb391 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #9  0x00007ffff7e1f763 in sycl::_V1::detail::enqueue_kernel_launch::handleOutOfResources(sycl::_V1::detail::device_impl const&, ur_kernel_handle_t_*, sycl::_V1::detail::NDRDescT const&) ()
      from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #10 0x00007ffff7e261ea in sycl::_V1::detail::enqueue_kernel_launch::handleErrorOrWarning(ur_result_t, sycl::_V1::detail::device_impl const&, ur_kernel_handle_t_*, sycl::_V1::detail::NDRDescT const&) ()
      from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #11 0x00007ffff7eeaa72 in sycl::_V1::detail::enqueueImpKernel(std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::NDRDescT&, std::vector<sycl::_V1::detail::ArgDesc, std::allocator<sycl::_V1::detail::ArgDesc> >&, std::shared_ptr<sycl::_V1::detail::kernel_bundle_impl> const&, std::shared_ptr<sycl::_V1::detail::kernel_impl> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::vector<ur_event_handle_t_*, std::allocator<ur_event_handle_t_*> >&, std::shared_ptr<sycl::_V1::detail::event_impl> const&, std::function<void* (sycl::_V1::detail::AccessorImplHost*)> const&, ur_kernel_cache_config_t, bool, bool, sycl::_V1::detail::RTDeviceBinaryImage const*) () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #12 0x00007ffff7f37cb5 in sycl::_V1::handler::finalize()::$_0::operator()() const ()
      from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #13 0x00007ffff7f34960 in sycl::_V1::handler::finalize() () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #14 0x00007ffff7ebd481 in void sycl::_V1::detail::queue_impl::finalizeHandler<sycl::_V1::handler>(sycl::_V1::handler&, sycl::_V1::event&) () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #15 0x00007ffff7ebc5e8 in sycl::_V1::detail::queue_impl::submit_impl(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, bool, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #16 0x00007ffff7ec16e8 in sycl::_V1::detail::queue_impl::submit(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #17 0x00007ffff7f63099 in sycl::_V1::queue::submit_impl(std::function<void (sycl::_V1::handler&)>, sycl::_V1::detail::code_location const&) () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #18 0x000000000040442d in sycl::_V1::queue::submit<main::{lambda(sycl::_V1::handler&)#1}>(main::{lambda(sycl::_V1::handler&)#1}, sycl::_V1::detail::code_location const&) (this=0x7fffffffb9a0, CGF=..., CodeLoc=...)
      at /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../include/sycl/queue.hpp:359
   #19 0x0000000000403f83 in main ()
      at 1_matrix_mul_SLM_size.cpp:104
   ```

4. Look at the final frame. (Your frame number might differ, and you might have to repeat this commend).
   ```
   (gdb) frame 19
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

Now exit the debugger.

#### Root-Cause the Issue

You can see that there is something wrong in the submit at line `104`, but we need more information to understand what is happening. For that we need to capture the lower-level API calls using the `unitrace` tool.

>**Note**: You must have already built the [Unified Tracing and Profiling Tool](#getting-the-tracing-and-profiling-tool). Once you have built the utility, you can invoke it before your program (similar to GBD).

Among other things, the Tracing and Profiling utility can print every low-level API call made to OpenCL™ or Level Zero. This is the feature that we will use to get more information about the crash.

2. Run the program with `unitrace` and enable the runtime debug messages:
   ```
   unitrace -c ./1_matrix_mul_SLM_size
   ```

3. Let the output continue until the error occurs and the program stops.
   ```
      :
   >>>> [1318232902142699] zeKernelSetGroupSize: hKernel = 51602008 groupSizeX = 10 groupSizeY = 1 groupSizeZ = 1
   <<<< [1318232902148340] zeKernelSetGroupSize [1253 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1318232902153233] zeCommandListCreateImmediate: hContext = 49997792 hDevice = 45298968 altdesc = 140727474880176 {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC(0xe) 0 0 0 0 2 0} phCommandList = 140727474880160 (hCommandList = 0)
   <<<< [1318232902418068] zeCommandListCreateImmediate [259479 ns] hCommandList = 51583896 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1318232902426142] zeEventHostReset: hEvent = 49925848
   <<<< [1318232902429796] zeEventHostReset [1372 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1318232911894375] zeCommandListAppendLaunchKernel: hCommandList = 51583896 hKernel = 51602008 (_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_) pLaunchFuncArgs = 140727474881304 {16385, 1, 1} hSignalEvent = 49925848 numWaitEvents = 0 phWaitEvents = 0
   <<<< [1318232911950754] zeCommandListAppendLaunchKernel [45519 ns] -> ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY(0x1879048195)
   terminate called after throwing an instance of 'sycl::_V1::exception'
     what():  UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
   Aborted (core dumped)
   ```

  **Clue**: By running the program under `unitrace` we can see that the error happens when launching a kernel called `(_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_`), and that this fails with an `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` error.

   A note about the output above. You will see that is has two lines that read:

   ```
   >>>> [1318232902142699] zeKernelSetGroupSize: hKernel = 51602008 groupSizeX = 10 groupSizeY = 1 groupSizeZ = 1
   :
   >>>> [1318232911894375] zeCommandListAppendLaunchKernel: hCommandList = 51583896 hKernel = 51602008 (_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_) pLaunchFuncArgs = 140727474881304 {16385, 1, 1} hSignalEvent = 49925848 numWaitEvents = 0 phWaitEvents = 0
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
:
computeUnits for each thread: 8192
perHwThreadPrivateMemorySize: 11776      totalPrivateMemorySize: 96468992
perHwThreadScratchSize: 65536    totalScratchSize: 536870912
perHwThreadPrivateScratchSize: 0         totalPrivateScratchSize: 0
Flush Task for Immediate command list : Enabled
Using PCI barrier ptr: 0x772ae3972000
Size of SLM (656384) larger than available (131072)
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
Aborted (core dumped)
```

The new message of interest is `Size of SLM (656384) larger than available (131072)`. This tells you that the size of the Shared Local Memory (SLM) memory on the device, 131072 bytes (128Kb), is smaller than the requested size of 656384 bytes (641Kb).

If the `parallel_for` were operating over a multi-dimensional range (for example, if `acc` were two or three-dimensional), you need to multiply the dimensions together to determine the number of floating point numbers we are trying to store in SLM. In our case, the calculation is easy:  the first argument to the `nd_range` in the `parallel_for` is single-dimensional, so it's just 163850. Thus the problem is that the size of work-group local memory we tried to allocate, (163850 floats or 4*163850=655,400 bytes rounded up to the nearest 64-byte cache line), doesn't fit in the SLM on this device.

You should know that different devices will have different amounts of memory set aside as SLM. In SYCL, you can query this number by passing `info::device::local_mem_size` to the `get_info` member of the `device` class.

Finally, running under `unitrace -c` you see:

```
:
>>>> [1318685383432507] zeKernelSetGroupSize: hKernel = 61899240 groupSizeX = 10 groupSizeY = 1 groupSizeZ = 1
<<<< [1318685383437033] zeKernelSetGroupSize [1000 ns] -> ZE_RESULT_SUCCESS(0x0)
>>>> [1318685383440955] zeCommandListCreateImmediate: hContext = 60295024 hDevice = 55597352 altdesc = 140733556795888 {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC(0xe) 0 0 0 0 2 0} phCommandList = 140733556795872 (hCommandList = 0)
Flush Task for Immediate command list : Enabled
Using PCI barrier ptr: 0xf7c9818f000
<<<< [1318685383694442] zeCommandListCreateImmediate [247603 ns] hCommandList = 61881128 -> ZE_RESULT_SUCCESS(0x0)
>>>> [1318685383702292] zeEventHostReset: hEvent = 60224120
<<<< [1318685383706720] zeEventHostReset [1427 ns] -> ZE_RESULT_SUCCESS(0x0)
>>>> [1318685383716840] zeCommandListAppendLaunchKernel: hCommandList = 61881128 hKernel = 61899240 (_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi1EEEE_) pLaunchFuncArgs = 140733556797016 {16385, 1, 1} hSignalEvent = 60224120 numWaitEvents = 0 phWaitEvents = 0
Size of SLM (656384) larger than available (131072)
<<<< [1318685383768499] zeCommandListAppendLaunchKernel [43385 ns] -> ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY(0x1879048195)
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  UR backend failed. UR backend returns:40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
Aborted (core dumped)
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

In real code, now that we have deduced the variable that is the source of the problem (the `acc` array in our synthetic code, which is defined as a `local_accessor`, meaning it will be stored in device shared-local memory), the "fix" is to rethink your algorithm. For example, can you break up `acc` into smaller sections that will fit in SLM and operate on them separately, one after the other?  You should also determine whether `acc` really needs to be in work-group local memory.

As noted in *Shared Local Memory* topic of the *[oneAPI GPU Optimization Guide
Developer Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/shared-local-memory.html)*, this really only makes sense when work-items need to share data and communicate with each other within a work-group.

In the synthetic code shown above, none of this is happening (each iteration is independent of every other since `i` is a SYCL `id` class, meaning that `i.get_id()` returns a unique index for each of the 163850 iterations). There is no reason why `acc` needs to be in work-group local memory. Instead, `acc` could be a normal `accessor` with a `device` target and a `read_write` access mode that would live in device global memory.

If data-sharing and communication between work-items is required, then you will need to find a way to break up the problem in such a way that the required SLM doesn't exceed device limits (for example, use a "window" or buffer into a larger array).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
