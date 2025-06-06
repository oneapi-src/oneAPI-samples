# `Guided Matrix Multiplication Exception` Sample

The `Guided Matrix Multiplication Exception` sample demonstrates a guided approach to debugging SYCL exceptions from incorrect use of the SYCL* API. It uses the Intel® oneAPI Base Toolkit (Base Kit) and several tools included in the Base Kit.

The sample code is a simple program that multiplies together two large matrices and verifies the results.

| Property              | Description
|:---                   |:---
| What you will learn   | How to use backtraces in the Intel® Distribution for GDB* to locate incorrect use of the SYCL API.
| Time to complete      | 50 minutes

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The two samples in this tutorial show examples of situations where the SYCL runtime provides an assert when it detects incorrect use of the SYCL API that is not caught at build time. Unfortunately, these runtime error checks are not comprehensive, so not getting an assert does not indicate correct code structure or practices.

Currently, SYCL asserts only tell you that an error was detected, but not where it resides in your code. To determine the location, you must run the program in the Intel® Distribution for GDB* with debug symbols enabled. Turning off optimization can also help.

When you encounter the assert and the program stops running, issuing a `backtrace` command in the debugger will show a call stack that gets you close to the SYCL API call that was written incorrectly.

You may want to consult the SYCL spec about argument order and allowed values to figure out what you did wrong in some cases.

The sample includes three different versions of some simple matrix multiplication code.

| File name                         | Description
|:---                               |:---
| `1_matrix_mul_null_pointer.cpp`   | This example shows the assert you get when a null pointer is passed to a SYCL memcpy statement
| `2_matrix_mul_multi_offload.cpp`  | This example shows the assert you get when you try to execute more than one offload statement in a SYCL `submit` lambda function
| `3_matrix_mul.cpp`                | A working version of the matrix multiply code that uses unified shared memory (`1_matrix_mul_null_pointer.cpp` and `2_matrix_mul_multi_offload.cpp` are broken versions of this code)

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                      | Ubuntu* 24.04 LTS
| Hardware                | GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler 2025.1 <br> Intel® Distribution for GDB* 2025.1


## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each workgroup)
- Explicit memory operations using Unified Shared Memory (USM)
- SYCL* kernels (including parallel_for function and explicit memory copies)
- SYCL* queues

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build and Run the `Guided Matrix Multiplication Exception` Programs

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
1. Build the programs.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
2. Run the programs.
   ```
   make run_all
   ```
   > **Note**: **The application will crash because of errors in the SYCL API calls.** This is expected behavior.

   For the broken null pointer version only, enter the following:
   ```
   make run_1_null_pointer
   ```
   For the broken multiple offload version only, enter the following:
   ```
   make run_2_multi_offload
   ```
   For the working version only, enter the following:
   ```
   make run_3
   ```
3. Clean the program. (Optional)
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

These instructions assume you have installed the Intel® Distribution for GDB* and have a basic working knowledge of GDB.

### Setting up to Debug on the GPU
To learn how setup and use Intel® Distribution for GDB*, see the *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/get-started-guide-linux/current/overview.html)*.  Additional setup instructions you should follow are at *[GDB-PVC debugger](https://dgpu-docs.intel.com/system-user-guides/DNP-Max-1100-userguide/DNP-Max-1100-userguide.html#gdb-pvc-debugger)* and *[Configuring Kernel Boot Parameters](https://dgpu-docs.intel.com/driver/configuring-kernel-boot-parameters.html)*.

Documentation on using the debugger in a variety of situations can be found at *[Debug Examples in Linux](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/tutorial-debugging-dpcpp-linux/current/overview.html)*

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see the *[Level Zero Specification Documentation - Introduction](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html)* and *[Intel® oneAPI Level Zero](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/intel-oneapi-level-zero.html)*.

### Fixing the Null Pointer Version

In `1_matrix_mul_null_pointer` a null pointer is passed to a SYCL `memcpy` statement.  The error message clearly tells you which `memcpy` statement caused the error.

```
   ./1_matrix_mul_null_pointer
   Initializing
   Computing
   Device: Intel(R) Data Center GPU Max 1550
   Device compute units: 512
   Device max work item size: 1024, 1024, 1024
   Device max work group size: 1024
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Exception caught at File: 1_matrix_mul_null_pointer.cpp | Function: main | Line: 95 | Column: 5
   terminate called after throwing an instance of 'sycl::_V1::exception'
   what():  NULL pointer argument in memory copy operation.
   Aborted (core dumped)
```

As an exercise, let's find this a debugger (any host debugger will work; however, we will use the Intel® Distribution for GDB*).

1. Start the Intel® Distribution for GDB* debugger and run the application within the debugger.
   ```
   gdb-oneapi ./1_matrix_mul_null_pointer
   (gdb) run
   ```
   When you get the error message `Debugging of GPU offloaded code is not enabled`, ignore it and answer `n` to the question `Quit anyway? (y or n)`

2. Notice the application failure. The error is the same message seen when we ran it outside the debugger.
   ```
   Exception caught at File: 1_matrix_mul_null_pointer.cpp | Function: main | Line: 95 | Column: 5
   terminate called after throwing an instance of 'sycl::_V1::exception'
     what():  NULL pointer argument in memory copy operation.

   Thread 1.1 "1_matrix_mul_nu" received signal SIGABRT, Aborted.
   ```
3. Run a `backtrace` to get a summary showing the rough location that triggered the assert.
   ```
   (gdb) backtrace
   ```
4. Notice in the results that the exception was triggered around line 95 (frame 9):
   ```
   #0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
   #1  __pthread_kill_internal (signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:78
   #2  __GI___pthread_kill (threadid=<optimized out>, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
   #3  0x00007ffff744527e in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
   #4  0x00007ffff74288ff in __GI_abort () at ./stdlib/abort.c:79
   #5  0x00007ffff78a5ff5 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x00007ffff78bb0da in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #7  0x00007ffff78a5a55 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #8  0x00007ffff78bb391 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #9  0x00007ffff7eb7ea0 in sycl::_V1::detail::queue_impl::memcpy(std::shared_ptr<sycl::_V1::detail::queue_impl> const&, void*, void const*, unsigned long, std::vector<sycl::_V1::event, std::allocator<sycl::_V1::event> > const&, bool, sycl::_V1::detail::code_location const&) () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #10 0x00007ffff7f62421 in sycl::_V1::queue::memcpy(void*, void const*, unsigned long, sycl::_V1::detail::code_location const&) () from /opt/intel/oneapi/compiler/2025.0/lib/libsycl.so.8
   #11 0x0000000000403dfa in main ()
      at 1_matrix_mul_null_pointer.cpp:95
   ```
5. Examine the last frame using the following (it may be different from the output above):
   ```
   (gdb) frame 11
   ```
   You may need to issue this command twice before you see output similar to the following example:
   ```
   #11 0x0000000000403dfa in main ()
      at 1_matrix_mul_null_pointer.cpp:95
   95          q.memcpy(dev_b, 0, N*P * sizeof(float));
   (gdb)
   ```
   Notice that in this case a `0` was passed as one of the pointers in the `memcpy`, which is clearly the error described in the exception. In a real application you will need to examine each of the input variables to `memcpy` using the gdb `print` command, and then trace the pointers back to where they were initialized (possibly in another source file).

6.  Exit the debugger using the `quit`command.


### Fixing the Multiple Offload Version

In the second version, the code attempts to execute more than one offload statement in a SYCL `submit` lambda function. To find the `submit` in question, we again need to use a debugger

1. Start the Intel® Distribution for GDB* debugger and run the application within the debugger.
   ```
   gdb-oneapi ./2_matrix_mul_multi_offload
   (gdb) run
   ```
   When you get the error message `Debugging of GPU offloaded code is not enabled`, ignore it and answer `n` to the question `Quit anyway? (y or n)`

2. The error is the same message seen when we ran it outside the debugger.
   ```
   terminate called after throwing an instance of 'sycl::_V1::exception'
   what():  Attempt to set multiple actions for the command group. Command group must consist of a single kernel or explicit memory operation.

   Thread 1.1 "2_matrix_mul_mu" received signal SIGABRT, Aborted.
   ```
   The exception talks about a “command group” and that only a single command group is allowed within a `submit`.  A command group is something like a `parallel_for` or a SYCL `memcpy` statement – it’s a language construct or function call that makes something happen on the device.  Only one action is allowed per `submit` construct.

3. Run a `backtrace` to get summary showing the rough location that triggered the assert.
   ```
   (gdb) backtrace
   ```

4. Notice in the results (which should look something like the following) that the exception (frame 8) was triggered around line 98 (frame 19):
   ```
   #0  __pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
   #1  __pthread_kill_internal (signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:78
   #2  __GI___pthread_kill (threadid=<optimized out>, signo=signo@entry=6) at ./nptl/pthread_kill.c:89
   #3  0x00007ffff744527e in __GI_raise (sig=sig@entry=6) at ../sysdeps/posix/raise.c:26
   #4  0x00007ffff74288ff in __GI_abort () at ./stdlib/abort.c:79
   #5  0x00007ffff78a5ff5 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x00007ffff78bb0da in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #7  0x00007ffff78a5a55 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #8  0x00007ffff78bb391 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #9  0x00007ffff7f076a0 in sycl::_V1::handler::memcpy(void*, void const*, unsigned long) ()
      from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #10 0x0000000000404ba2 in main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const (
      this=0x7fffffffb2d8, h=sycl::handler& = {...})
      at /nfs/site/home/cwcongdo/oneAPI-samples-true/Tools/ApplicationDebugger/guided_matrix_mult_Exceptions/src/2_matrix_mul_multi_offload.cpp:100
   #11 0x0000000000404b3d in std::__invoke_impl<void, main::{lambda(auto:1&)#1}&, sycl::_V1::handler&>(std::__invoke_other, main::{lambda(auto:1&)#1}&, sycl::_V1::handler&) (__f=..., __args=sycl::handler& = {...})
      at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/invoke.h:61
   #12 0x0000000000404add in std::__invoke_r<void, main::{lambda(auto:1&)#1}&, sycl::_V1::handler&>(main::{lambda(auto:1&)#1}&, sycl::_V1::handler&) (__fn=..., __args=sycl::handler& = {...})
      at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/invoke.h:111
   #13 0x00000000004049f5 in std::_Function_handler<void (sycl::_V1::handler&), main::{lambda(auto:1&)#1}>::_M_invoke(std::_Any_data const&, sycl::_V1::handler&) (__functor=..., __args=sycl::handler& = {...})
      at /usr/lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/std_function.h:290
   #14 0x00007ffff7e83121 in sycl::_V1::detail::queue_impl::submit_impl(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, bool, sycl::_V1::detail::code_location const&, bool, sycl::_V1::detail::SubmissionInfo const&) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #15 0x00007ffff7e895c8 in sycl::_V1::detail::queue_impl::submit_with_event(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::SubmissionInfo const&, sycl::_V1::detail::code_location const&, bool) () from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #16 0x00007ffff7f33afa in sycl::_V1::queue::submit_with_event_impl(std::function<void (sycl::_V1::handler&)>, sycl::_V1::detail::SubmissionInfo const&, sycl::_V1::detail::code_location const&, bool) ()
      from /opt/intel/oneapi/compiler/2025.1/lib/libsycl.so.8
   #17 0x00000000004048b3 in sycl::_V1::queue::submit_with_event<main::{lambda(auto:1&)#1}>(main::{lambda(auto:1&)#1}, sycl::_V1::queue*, sycl::_V1::detail::code_location const&) (this=0x7fffffffb860, CGF=..., SecondaryQueuePtr=0x0,
      CodeLoc=...) at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/queue.hpp:2826
   #18 0x00000000004042cd in sycl::_V1::queue::submit<main::{lambda(auto:1&)#1}>(main::{lambda(auto:1&)#1}, sycl::_V1::detail::code_location const&) (this=0x7fffffffb860, CGF=..., CodeLoc=...)
      at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/queue.hpp:365
   #19 0x0000000000403edc in main ()
      at 2_matrix_mul_multi_offload.cpp:98

   ```

5. Examine the last frame (it may be different from the output above) using the following command:
   ```
   (gdb) frame 19
   ```
   You may need to issue this command twice before you see output similar to the following example:
   ```
   #19 0x0000000000403e7c in main ()
       at 2_matrix_mul_multi_offload.cpp:98
   98          q.submit([&](auto &h) {
   ```

6. Examine the code.
   ```
   (gdb) list
   ```
   Notice that the `submit` in question is the following:
   ```
   97          // Submit command group to queue to initialize matrix c
   98          q.submit([&](auto &h) {
   99              h.memcpy(dev_c, &c_back[0], M*P * sizeof(float));
   100             h.memcpy(dev_c, &c_back[0], M*P * sizeof(float));
   101         });
   ```

   As the exception reported, we are trying to do two memory copies to the device within the `submit` statement, where only a single `parallel_for` or `memcpy` is allowed.

7. To fix the error, remove the extra `memcpy` from the code above.   If this statement were actually issuing two different `memcpy` statements, you would update the code to break this up into two `submit` statements, each with a single `memcpy` .
   ```
    q.submit([&](auto &h) {
        h.memcpy(<arguments for first memcpy>);
    });
    q.submit([&](auto &h) {
        h.memcpy(<arguments for second memcpy>);
    });
   ```

8.  Exit the debugger using the `quit` command.

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
