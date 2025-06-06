# `Guided Matrix Multiplication Race Condition` Sample

The `Guided Matrix Multiplication Race Condition` sample demonstrates a guided approach to debugging a race condition accessing data on the host before it has been fully copied back from the device. It uses the Intel® oneAPI Base Toolkit (Base Kit) and several tools included in the Base Kit.

The sample is a simple program that multiplies together two large matrices and verifies the results.

| Property               | Description
|:---                    |:---
| What you will learn    | A way to root-cause incorrect use of the SYCL language.
| Time to complete       | 50 minutes

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The sample in this tutorial shows how to root-cause incorrect use of the SYCL language:  accessing data on the host before it has been fully copied back from the device.

This example results in a race condition that really doesn't give any clue as to the nature of the problem.  We will show you a suite of techniques that might help you find a similar problem in more complex code.

The sample includes different versions of a simple matrix multiplication program.

| File name                           | Description
|:---                                 |:---
| `1_matrix_mul_race_condition.cpp`   |This example shows what happens when a developer tries to access data provided by the device before the copy to the host is complete.
| `2_matrix_mul.cpp`                  | A working version of the matrix multiply code that properly waits for the data to be copied back to the host.
| `3_matrix_mul.cpp`                  | A working version of the application that corrects its errors using a host accessor and a `q.wait` command in place of parenthesis.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 24.04 LTS
| Hardware                | GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler 2025.1 <br> Intel® Distribution for GDB* 2025.1 <br> Unified Tracing and Profiling Tool 2.1.2, which is available from the [following Github repository](https://github.com/intel/pti-gpu/tree/master/tools/unitrace).
| Intel GPU Driver | Intel® General-Purpose GPU Rolling Release driver 2507.12 or later from https://dgpu-docs.intel.com/releases/releases.html

## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each workgroup)
- SYCL* kernels
- SYCL* queues

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build and Run the `Guided Matrix Multiply Race Condition` Programs

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
3. Run the programs.
   ```
   make run_all
   ```
   >**Note**: **The application will crash because of errors in the SYCL API calls.** This is expected behavior.

   For the broken version only, enter the following:
   ```
   make run_bugged
   ```
   For the intermediate working version only, enter the following:
   ```
   make run_2
   ```
   For the fully corrected version only, enter the following:
   ```
   make run_3
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

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html)* for more information on using the utility.


## Guided Debugging

This example shows what happens when code tries to access data provided by the device before the copy to the host is complete.

These instructions assume you have installed the Intel® Distribution for GDB* and have a basic working knowledge of GDB.

### Setting up to Debug on the GPU
To learn how setup and use Intel® Distribution for GDB*, see the *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/get-started-guide-linux/current/overview.html)*.  Additional setup instructions you should follow are at *[GDB-PVC debugger](https://dgpu-docs.intel.com/system-user-guides/DNP-Max-1100-userguide/DNP-Max-1100-userguide.html#gdb-pvc-debugger)* and *[Configuring Kernel Boot Parameters](https://dgpu-docs.intel.com/driver/configuring-kernel-boot-parameters.html)*.

Documentation on using the debugger in a variety of situations can be found at *[Debug Examples in Linux](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/tutorial-debugging-dpcpp-linux/current/overview.html)*

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see the *[Level Zero Specification Documentation - Introduction](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html)* and *[Intel® oneAPI Level Zero](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/intel-oneapi-level-zero.html)*.

### Getting the Tracing and Profiling Tool

At a step in this tutorial, the instructions require a utility that was not installed with the Intel® oneAPI Base Toolkit (Base Kit).

To complete the steps in the following section, you must download the [Unified Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) code from GitHub and build the utility. The build instructions are included in the README in the GitHub repository.  This build will go much more smoothly if you first install the latest drivers from [the Intel GPU driver download site](https://dgpu-docs.intel.com/driver/overview.html), especially the development packages (only available in the Data Center GPU driver install ).  Once you have built the utility, you invoke it on the command line in front of your program (similar to using GDB).

### Examine the Original Code

As you might have noticed, when you attempt to run `1_matrix_mul_race_condition.cpp` the code reports bad results and then exits. We can use the Intel® Distribution for GDB* to get a backtrace of the entire stack to understand the problem.  

In case we need view code running on the GPU, we need to enable GPU debugging.  This will require [some setup on your system](#setting-up-to-debug-on-the-gpu) before you can see code running on the GPU.

1. Run the Intel® Distribution for GDB*.  
   ```
   ZET_ENABLE_PROGRAM_DEBUGGING=1 gdb-oneapi 1_matrix_mul_race_condition
   ```
2. Then run the application in the debugger.
   ```
   run
   ```
3. Examine the results.
   ```
   Device: Intel(R) Data Center GPU Max 1550
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Result of matrix multiplication using DPC++: Fail - The result is incorrect for element: [0, 0], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 1], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 2], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 3], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 4], expected: 45150, but found: 0
   Fail - The results mismatch!
   [Thread 0x7fffe40006c0 (LWP 88450) exited]
   [Inferior 1 (process 88423) exited with code 0377]
   Detached from device [0000:4d:00.0]
   [Inferior 2 (device [0000:4d:00.0]) detached]
   Detached from device [0000:4d:00.0]
   [Inferior 3 (device [0000:4d:00.0]) detached]
   intelgt: inferior 2 (gdbserver-ze) has been removed.
   intelgt: inferior 3 (gdbserver-ze) has been removed.
   (gbd)
   ```
   As we saw outside the debugger, it ran to completion.   But note that the inferior (the code running on the GPU), exited with an error code (0377).   Let's see if we can trap that error.

4. Run again, telling the debugger to stop if the application throws an exception
   ```
   (gdb) catch throw
   Catchpoint 1 (throw)
   (gdb) run
   ```
   We run, and stop with this:
   ```
   Device: Intel(R) Data Center GPU Max 1550
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Result of matrix multiplication using DPC++: Fail - The result is incorrect for element: [0, 0], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 1], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 2], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 3], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 4], expected: 45150, but found: 0
   Fail - The results mismatch!
   [Switching to thread 1.1 (Thread 0x7ffff7acdf00 (LWP 88464))]

   Thread 1.1 "1_matrix_mul_ra" hit Catchpoint 1 (exception thrown), 0x00007ffff78bb35a in __cxa_throw ()
      from /lib/x86_64-linux-gnu/libstdc++.so.6
   (gdb)
   ```

5. Use `backtrace` to see where we ended up.
   ```  
   (gdb) backtrace
   ```
   The output might look similar to the following:
   ```
   #0  0x00007ffff78bb35a in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
   #1  0x00007ffff7cbebf4 in void sycl::_V1::detail::Adapter::checkUrResult<(sycl::_V1::errc)1>(ur_result_t) const () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #2  0x00007ffff7e2abde in sycl::_V1::detail::copyD2H(sycl::_V1::detail::SYCLMemObjI*, ur_mem_handle_t_*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, char*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, std::vector<ur_event_handle_t_*, std::allocator<ur_event_handle_t_*> >, ur_event_handle_t_*&, std::shared_ptr<sycl::_V1::detail::event_impl> const&) ()
      from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #3  0x00007ffff7e2b73c in sycl::_V1::detail::MemoryManager::copy(sycl::_V1::detail::SYCLMemObjI*, void*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, void*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, std::vector<ur_event_handle_t_*, std::allocator<ur_event_handle_t_*> >, ur_event_handle_t_*&, std::shared_ptr<sycl::_V1::detail::event_impl> const&) ()
      from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #4  0x00007ffff7eb31e9 in ur_result_t sycl::_V1::detail::callMemOpHelper<void (sycl::_V1::detail::SYCLMemObjI*, void*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, void*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, std::vector<ur_event_handle_t_*, std::allocator<ur_event_handle_t_*> >, ur_event_handle_t_*&, std::shared_ptr<sycl::_V1::detail::event_impl> const&), sycl::_V1::detail::SYCLMemObjI*, void*, std::shared_ptr<sycl::_V1::detail::queue_impl>&, unsigned int&, sycl::_V1::range<3>&, sycl::_V1::range<3>&, sycl::_V1::id<3>&, unsigned int&, void*&, std::shared_ptr<sycl::_V1::detail::queue_impl>&, unsigned int&, sycl::_V1::range<3>&, sycl::_V1::range<3>&, sycl::_V1::id<3>&, unsigned int&, std::vector<ur_event_handle_t_*, std::allocator<ur_event_handle_t_*> >, ur_event_handle_t_*&, std::shared_ptr<sycl::_V1::detail::event_impl>&>(void (&)(sycl::_V1::detail::SYCLMemObjI*, void*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, void*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned int, sycl::_V1::range<3>, sycl::_V1::range<3>, sycl::_V1::id<3>, unsigned int, std::vector<ur_event_handle_t_*, std::allocator<ur_event_handle_t_*> >, ur_event_handle_t_*&, std::shared_ptr<sycl::_V1::detail::event_impl> const&), sycl::_V1::detail::SYCLMemObjI*&&, void*&&, std::shared_ptr<sycl::_V1::detail::queue_impl>&, unsigned int&, sycl::_V1::range<3>&, sycl::_V1::range<3>&, sycl::_V1::id<3>&, unsigned int&, void*&, std::shared_ptr<sycl::_V1::detail::queue_impl>&, unsigned int&, sycl::_V1::range<3>&, sycl::_V1::range<3>&, sycl::_V1::id<3>&, unsigned int&, std::vector<ur_event_handle_t_*, std::allocator<ur_event_handle_t_*> >&&, ur_event_handle_t_*&, std::shared_ptr<sycl::_V1::detail::event_impl>&) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #5  0x00007ffff7eb2c4e in sycl::_V1::detail::MemCpyCommandHost::enqueueImp() () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #6  0x00007ffff7ea90fb in sycl::_V1::detail::Command::enqueue(sycl::_V1::detail::EnqueueResultT&, sycl::_V1::detail::BlockingT, std::vector<sycl::_V1::detail::Command*, std::allocator<sycl::_V1::detail::Command*> >&) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #7  0x00007ffff7ecec2e in sycl::_V1::detail::Scheduler::GraphProcessor::enqueueCommand(sycl::_V1::detail::Command*, std::shared_lock<std::shared_timed_mutex>&, sycl::_V1::detail::EnqueueResultT&, std::vector<sycl::_V1::detail::Command*, std::allocator<sycl::_V1::detail::Command*> >&, sycl::_V1::detail::Command*, sycl::_V1::detail::BlockingT) ()
      from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #8  0x00007ffff7eca5ba in sycl::_V1::detail::Scheduler::addCopyBack(sycl::_V1::detail::AccessorImplHost*) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #9  0x00007ffff7edd2e6 in sycl::_V1::detail::SYCLMemObjT::updateHostMemory(void*) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #10 0x00007ffff7eebdd3 in std::_Function_handler<void (std::function<void (void*)> const&), sycl::_V1::detail::SYCLMemObjT::handleHostData(void*, unsigned long)::{lambda(std::function<void (void*)> const&)#1}>::_M_invoke(std::_Any_data const&, std::function<void (void*)> const&) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #11 0x00007ffff7eeb3ea in std::_Function_handler<void (), sycl::_V1::detail::SYCLMemObjT::set_final_data(std::function<void (std::function<void (void*)> const&)> const&)::{lambda()#1}>::_M_invoke(std::_Any_data const&) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #12 0x00007ffff7edd558 in sycl::_V1::detail::SYCLMemObjT::updateHostMemory() () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #13 0x00007ffff7eeb8b4 in std::_Sp_counted_ptr_inplace<sycl::_V1::detail::buffer_impl, std::allocator<sycl::_V1::detail::buffer_impl>, (__gnu_cxx::_Lock_policy)2>::_M_dispose() ()
      from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #14 0x00000000004097ea in std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release (this=0x3923630)
      at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/shared_ptr_base.h:346
   #15 0x0000000000409766 in std::__shared_count<(__gnu_cxx::_Lock_policy)2>::~__shared_count (this=0x7fffffffb480)
      at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/shared_ptr_base.h:1071
   #16 0x000000000040bc09 in std::__shared_ptr<sycl::_V1::detail::buffer_impl, (__gnu_cxx::_Lock_policy)2>::~__shared_ptr (this=0x7fffffffb478)
      at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/shared_ptr_base.h:1524
   #17 0x000000000040bbe5 in std::shared_ptr<sycl::_V1::detail::buffer_impl>::~shared_ptr (this=0x7fffffffb478)
      at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/shared_ptr.h:175
   #18 0x000000000040b105 in sycl::_V1::detail::buffer_plain::~buffer_plain (this=0x7fffffffb478) at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/buffer.hpp:87
   #19 0x0000000000409512 in sycl::_V1::buffer<float, 2, sycl::_V1::detail::aligned_allocator<float>, void>::~buffer (this=0x7fffffffb478)
      at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/buffer.hpp:485
   #20 0x0000000000403c96 in main () at 1_matrix_mul_race_condition.cpp:129
   (gdb)
   ````
6. Look at the final frame. (Your frame number might differ.)
    ```
    (gdb) frame 20
    ```
   You should see something similar to the following:
    ```
    #19 0x0000000000404dbe in main () at 1_matrix_mul_race_condition.cpp:129
    129     }
    ```
7. Examine the code in that region.
    ```
    (gdb) list
    ```
   The output should be similar to the following:
    ```
    124       cout << "Result of matrix multiplication using DPC++: ";
    125       result = VerifyResult(c_back);
    126       delete[] c_back;
    127
    128       return result;
    129     }
    130
    131     bool ValueSame(float a, float b) {
    132       return fabs(a - b) < numeric_limits<float>::epsilon();
    133     }
    ```

8. Exit the debugger.

9. Run the program using the [Unified Tracing and Profiling Tool](#getting-the-tracing-and-profiling-tool) tool.

   >**Note**: You must modify the command shown below to include the path to where you installed the `unitrace` utility.  You also may need to source the oneAPI environment using `source /opt/intel/oneapi/setvars.sh`.
    ```
    [path]/unitrace -c ./1_matrix_mul_race_condition
    ```
    The very last lines of the output will be something like this:
    ```
   <<<< [211022912597111] zeCommandListAppendLaunchKernel [22946 ns] hWaitEvents = 37891702 -> ZE_RESULT_SUCCESS(0x0)
   Result of matrix multiplication using DPC++: Fail - The result is incorrect for element: [0, 0], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 1], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 2], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 3], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 4], expected: 45150, but found: 0
   Fail - The results mismatch!
   >>>> [211022976795378] zeEventCreate: hEventPool = 37983040 desc = 140734297023408 {ZE_STRUCTURE_TYPE_EVENT_DESC(0x11) 0 4 4 0} phEvent = 140734297023464 (hEvent = 15669694584003)
   <<<< [211022976802066] zeEventCreate [1339 ns] hEvent = 37883384 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [211022976805342] zeCommandListAppendMemoryCopyRegion: hCommandList = 37277672 dstptr = 35936816 dstRegion = 140734297023744 dstPitch = 2400 dstSlicePitch = 360000 srcptr = 18374967954634571776 srcRegion = 140734297023768 srcPitch = 2400 srcSlicePitch = 360000 hSignalEvent = 37883384 numWaitEvents = 1 phWaitEvents = 37993232 (hWaitEvents = [37882840])
   <<<< [211022976856153] zeCommandListAppendMemoryCopyRegion [46127 ns] hWaitEvents = 37882840 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [211022976862571] zeEventHostSynchronize: hEvent = 37883384 timeout = 18446744073709551615
   <<<< [211022979801501] zeEventHostSynchronize [2937354 ns] -> ZE_RESULT_SUCCESS(0x0)
   Segmentation fault (core dumped)
    ```

### Interpret the Results

The first clue here is that the program throws an exception after it has completed checking the results and finding them bad. That behavior is worrying.

Next, looking at the crash in the debugger, there are a couple of odd things that stand out.   Look at stack `frame 9`.  This frame shows us attempting to update the host memory from the device, while `frame 20` shows we are already at the end of the program and have started cleaning up the SYCL buffers (`frame 19`).  The only variable containing data returned from the device is `c_back`.  But the developer has already deleted `c_back` in line 126, so the *data the buffer being copied into (`c_back`) no longer exists*.

We see something like this in the `unitrace` output above.   The kernel is executed, the results are immediately checked, we create and wait on some events, and then the last thing we try to do before crashing is to copy some memory from the device memory (`srcptr = 18374967954634571776`) to a host pointer (`dstptr = 35936816`) that previously was used to initialize this same device memory (around line 101).   Since `c_buf` is the only accessor that is defined as writeable in the `q.submit` at line 97, it again is a likely suspect.  

But what if the developer didn't delete `c_back`, and let program termination clean it up?  Try it!  Unfortunately, in that case your program complains about bad results, but it exits cleanly (shutdown will wait for the GPU to copy memory back to the host buffer before it kills the buffer).

Is the behavior different if you run it on OpenCL or Level 0?  The default is to use the Level Zero run time, but we can explicitly force the use of either Level Zero or OpenCL, which can be helpful when troubleshooting.

```
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./1_matrix_mul_race_condition
ONEAPI_DEVICE_SELECTOR=opencl:gpu ./1_matrix_mul_race_condition
```
Unfortunately not; pretty much the same thing happens - they both produce incorrect results on exiting.

> **Note:** the command with OpenCL will only work if the `sycl-ls` command shows OpenCL
> devices for the graphics card, such as like this:

   ```
   $ sycl-ls
   [opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz OpenCL 3.0 (Build 0) [2024.18.6.0.02_160000]
   [opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [24.22.29735.27]
   [opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [24.22.29735.27]
   [opencl:cpu][opencl:3] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz OpenCL 3.0 (Build 0) [2023.16.7.0.21_160000]
   [opencl:fpga][opencl:4] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.7.0.21_160000]
   [level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 1.3 [1.3.29735]
   [level_zero:gpu][level_zero:1] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 1.3 [1.3.29735]
   ```

   If you are missing `[opencl:gpu]` devices you may have to add the necessary libraries to your device path by setting the appropriate path in `DRIVERLOC` and then running the following four commands (for Ubuntu - adapt for other OSes):

   ```
   export DRIVERLOC=/usr/lib/x86_64-linux-gnu
   export OCL_ICD_FILENAMES=$OCL_ICD_FILENAMES:$DRIVERLOC/intel-opencl/libigdrcl.so
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DRIVERLOC
   export PATH=$PATH:/opt/intel/oneapi:$DRIVERLOC
   ```

Similarly, we specify targeting the CPU, which sometimes can avoid problems in your code that are specific to offloading to the GPU.
```
ONEAPI_DEVICE_SELECTOR=*:cpu ./1_matrix_mul_race_condition
```
This also has problems, but if you run this in the debugger you will see lots of threads all running in the third `q.submit` kernel, but no thread running `main`.   This is because these threads have been abandoned when `main` deleted `c_back` and exited!

So in conclusion, it looks like the third kernel is still executing and/or its results are still being copied back to the host as the program is terminating.  Which explains the incorrect results (they aren't available on the host yet) and the crash (results from the card are being copied to memory that has been deallocated).  All these point to some sort of synchronization issue or race condition between the host and device.

### Understand the Problem

Because we are using SYCL buffers, even though the `q.submit` statements that populate `a_buf` and `b_buf` execute asynchronously, the third `q.submit` statement does not execute until those first two submits are complete because the SYCL runtime realizes that the third `q.submit` depends on the `a_buf` and `b_buf` buffers, which are being used in the first two kernels.   Once the first two kernels complete, the third `q.submit` kernel starts executing because both its inputs are ready. The SYCL runtime then immediately returns control to the host and we proceed to the code which verifies the result - ***while the third `q.submit` keeps running***.

There are three errors in this code:

1. As just noted, we did not wait until the third `q.submit` kernel completed before accessing the data in `c_back`. This could either be done using parenthesis to enforce scope, and thus order of operations, or by adding a `q.wait()` call just before the call to `VerifyResult`.

2. We  should be using a host accessor pointing to SYCL buffer `c_buf` to access its contents, which would also indicate that we need to wait for the third `q.submit kernel` to complete **and** for the *data to be copied back to the host* before accessing the data in `c_back`.

3. For buffers initialized with a pointer to host memory (like `c_buf`), the developer "makes a contract with the SYCL runtime" to not reference the host pointer again until the SYCL buffer is destroyed.  Thus, deleting the host memory before the SYCL buffer is destroyed is illegal (the call to `delete[] c_back;`). The buffer cannot detect that the memory was deallocated.

### Fix the Code

Let's fix one of these bugs at a time.

To address (1), go back into the program and add a `q.wait()` before the call to `VerifyResult`:

```
q.wait();

int result;
cout << "Result of matrix multiplication using DPC++: ";
result = VerifyResult(c_back);
delete[] c_back;
```

To address (2), we need to signal the SYCL runtime that the results of the kernel are needed back on the host for our `VerifyResult` call. We do this using a host accessor.

So, go back into the program and add a host accessor used to read `c_buf`, and update the forward definition and definition of `VerifyResult` as follows:

```
/**
* Perform matrix multiplication on host to verify results from device.
*/
int VerifyResult(sycl::host_accessor<float, 2, sycl::access::mode::read> result);

int main() {
:
:
q.wait();

int result;
host_accessor my_results(c_buf, read_only);
cout << "Result of matrix multiplication using DPC++: ";
result = VerifyResult(my_results);

  return result;
}

bool ValueSame(float a, float b) {
 return fabs(a - b) < numeric_limits<float>::epsilon();
}

int VerifyResult(sycl::host_accessor<float, 2, sycl::access::mode::read> c_back) {
// Check that the results are correct by comparing with host computing.
int i, j, k;
:
```
The result should look like `3_matrix_mul.cpp`.  Reiterating, with these changes :
1.  We created a host accessor to pull the values of out `c_buf` on the host, forcing the data to be transferred from the device to the host before the first access (one of the race conditions in this code).
2.  We waited for the third `q.submit` kernel to complete before asking for the values in `c_buf`, fixing the other race condition.
3.  We are no longer deleting `c_back` before the SYCL buffer that makes use of it (`c_buf`) is destroyed on program exit.
4.  We changed `VerifyResult` to pass down the host accessor, with which we are able to read the contents of the accessor the same way we would access the original `c_back` array (which we "made a contract" not to look at while a SYCL buffer was making use of it).

### Examine the Problem from a Different Perspective

Compare `1_matrix_mul_race_condition.cpp` and `2_matrix_mul.cpp` source files.

Note that the source files differ by **two characters** (parentheses) only.

```
54d53
<   {
123d121
<   }
```
This is a more detailed look at this region of the code.
```
  {   // This is unique to 2_matrix_mul.cpp
    queue q(default_selector{});
    :
    q.submit([&](auto &h) {
      :
      // Execute kernel.
      h.parallel_for(range(M, P), [=](auto index) {
        :
        accessor c(c_buf, h, write_only);
        :
        // Compute the result of one element of c
        for (int i = 0; i < width_a; i++) {
          sum += a[row][i] * b[i][col];
        }

        c[index] = sum;
      });
    });
  } // This is unique to 2_matrix_mul.cpp

  int result;
  cout << "Result of matrix multiplication using SYCL: ";
  result = VerifyResult(c_back);
```
If you run this version of the code, you will see that it completes correctly.

Realizing that in the real world we may not have such simple source at hand, let's look a little more closely.  The extra brackets in `2_matrix_mul.cpp` start just before the SYCL queue is defined, and end before we look at the results in `c_back`.

As a result, because the device queue `q` exists only in the scope of those brackets, host execution waits at the close bracket until the third kernel completes and the updates made to the write-only accessor `c` pointing to `c_buf` get copied from the device and into `c_back`.

Note that `2_matrix_mul.cpp` still has a bug.  It is an example of problem (2) above - it's not using a host accessor to access the data in `c_back`, which is still being managed by SYCL buffer `c_buf`.   This violates the contract (3 above) that we are not allowed to look at the host data while it is being managed by a SYCL buffer.  We just got lucky.

This points out a potential trap in the training documentation you may have read while learning SYCL.   You can easily get the impression that if you use the SYCL buffer-accessor mechanism, synchronization will be taken care of for you.  The use of parenthesis may be mentioned in passing with little explanation.   Even though the documentation may say "the { } block ensures all SYCL work has concluded," this is not stressed.

This is the trap of the SYCL buffer-accessor mechanism - you may assume that the automatic synchronization mechanism is smarter than it really is.  In `1_matrix_mul_race_condition.cpp`, the SYCL runtime does not realize that we cannot call `VerifyResult` with the `c_back` array until the third `q.submit` kernel completes and the data are copied back to the host - it assumes you know what you are doing.

>**Note**: You will find more on the proper use of buffers and accessors in the *Buffer Accessor Mode* section of the *[oneAPI GPU Optimization Guide Developer Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/buffer-accessor-modes.html)*.

### Generalized Approach

As we saw, `1_matrix_mul_race_condition.cpp` suffered from multiple race conditions and from directly manipulating data contained in a SYCL buffer without informing the buffer (this was an extreme case - we deleted the underlying data managed by the buffer).

But this gives us some hints on how to find these types of problems in any sort of code.

1.  The kernel *is* producing correct results, but they are not getting to the host when the host tries to access them.
2.  The host crashes when your host code calls a buffer destructor (frames 15-19 above).
3.  The host crashes when the SYCL runtime attempts to copy device data to a buffer on the host (frame 9 above).

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
