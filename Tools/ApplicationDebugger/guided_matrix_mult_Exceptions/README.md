# `Guided Matrix Multiplication Exception` Sample

The `Guided Matrix Multiplication Exception` sample demonstrates an approach to debugging SYCL exceptions from incorrect use of the SYCL* API using several tools in Intel® oneAPI.

The sample code is a simple program that multiplies together two large matrices and verifies the results.

| Property              | Description
|:---                   |:---
| What you will learn   | How to use backtraces in the Intel® Distribution for GDB* to locate incorrect use of the SYCL API.
| Time to complete      | 50 minutes

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The two samples in this tutorial show situations where the SYCL runtime provides an assert when it detects incorrect use of the SYCL API that is not caught at build time. Unfortunately, these runtime error checks are not comprehensive, so not getting an assert does not indicate correct code structure or practices.

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

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 24.04 LTS
| Intel Graphics Hardware | GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler 2026.0 <br> Intel® Distribution for GDB* 2026.0
| Intel GPU Driver | Intel® General-Purpose GPU Long-Term Support driver 2523.59 or later from https://dgpu-docs.intel.com/releases/releases.html


## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each workgroup)
- Explicit memory operations using Unified Shared Memory (USM)
- SYCL* kernels (including parallel_for function and explicit memory copies)
- SYCL* queues

## Set Environment Variables

When working with the command-line interface (CLI), set up your oneAPI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries and tools are ready for development.

## Build and Run the `Guided Matrix Multiplication Exception` Programs

> **Note**: If you have not already done so, set up your CLI
environment by sourcing  the `setvars` script in the root of your oneAPI installation.
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


## Guided Debugging

These instructions assume you have installed the Intel® Distribution for GDB* and have a basic working knowledge of GDB.

### Setting up to Debug on the GPU
To learn how setup and use Intel® Distribution for GDB*, see the *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/get-started-guide-linux/current/overview.html)*.  Additional setup instructions you should follow are at *[GPU Debugging](https://dgpu-docs.intel.com/driver/gpu-debugging.html)* and *[Configuring Kernel Boot Parameters](https://dgpu-docs.intel.com/driver/configuring-kernel-boot-parameters.html)*.

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
   Exception caught at File: 1_matrix_mul_null_pointer.cpp | Function: main | Line: 95 | Column: 7
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
   > When you get the error message `Debugging of GPU offloaded code is not enabled`, ignore it and answer `n` to the question `Quit anyway? (y or n)`.  You may need to do this more than once.   

   > Why can we ignore these messages and keep on debugging anyway?  Because we don't need to monitor the code running on the device in the debugger - the asserts are coming from the host during the call of the kernel.  Running `gdb-oneapi` with `ZET_ENABLE_PROGRAM_DEBUGGING=1` is only necessary if you want to debug the kernels running on the GPU.

2. Notice the application failure. The error is the same message seen when we ran it outside the debugger.
   ```
   Exception caught at File: 1_matrix_mul_null_pointer.cpp | Function: main | Line: 95 | Column: 7
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
   > When you get the error message `Debugging of GPU offloaded code is not enabled`, ignore it and answer `n` to the question `Quit anyway? (y or n)`.  You may need to do this more than once.   

   > Why can we ignore these messages and keep on debugging anyway?  Because we don't need to monitor the code running on the device in the debugger - the asserts are coming from the host during the call of the kernel.  Running `gdb-oneapi` with `ZET_ENABLE_PROGRAM_DEBUGGING=1` is only necessary if you want to debug the kernels running on the GPU.

2. The error is the same message seen when we ran it outside the debugger.
   ```
   terminate called after throwing an instance of 'sycl::_V1::exception'
   what():  Attempt to set multiple actions for the command group. Command group must consist of a single kernel or explicit memory operation.

   Thread 1.1 "2_matrix_mul_mu" received signal SIGABRT, Aborted.
   ```
   The exception talks about a “command group” and that only a single command group is allowed within a `submit`.  A command group is something like a `parallel_for` or a SYCL `memcpy` statement – it’s a language construct or function call that makes something happen on the device.  Only one such action is allowed per `submit` construct.

3. Run a `backtrace` to get summary showing the rough location that triggered the assert.
   ```
   (gdb) backtrace
   ```

4. Notice in the results (which should look something like the following) that the exception (frame 8) was triggered around line 98 (frame 16):
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
   #9  0x00007ffff7f13f61 in sycl::_V1::handler::memcpy(void*, void const*, unsigned long) ()
      from /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9
   #10 0x0000000000404762 in main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const (
      this=0x7fffffffb318, h=sycl::handler& = {...})
      at Tools/ApplicationDebugger/guided_matrix_mult_Exceptions/src/2_matrix_mul_multi_offload.cpp:100
   #11 0x00000000004046fd in sycl::_V1::detail::type_erased_cgfo_ty::invoker<main::{lambda(auto:1&)#1}>::call(void const*, sycl::_V1::handler&) (object=0x7fffffffb318, cgh=sycl::handler& = {...})
      at /opt/intel/oneapi/compiler/2026.0/bin/compiler/../../include/sycl/handler.hpp:191
   #12 0x00007ffff7e92c6d in sycl::_V1::detail::queue_impl::submit_impl(sycl::_V1::detail::type_erased_cgfo_ty const&, bool, sycl::_V1::detail::code_location const&, bool, sycl::_V1::detail::_V1::SubmissionInfo const&) ()
      from /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9
   #13 0x00007ffff7f58ac9 in sycl::_V1::queue::submit_with_event_impl(sycl::_V1::detail::type_erased_cgfo_ty const&, sycl::_V1::detail::_V1::SubmissionInfo const&, sycl::_V1::detail::code_location const&, bool) const ()
      from /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9
   #14 0x00000000004072be in sycl::_V1::queue::submit_with_event<sycl::_V1::ext::oneapi::experimental::properties<sycl::_V1::ext::oneapi::experimental::detail::properties_type_list<> > >(sycl::_V1::ext::oneapi::experimental::properties<sycl::_V1::ext::oneapi::experimental::detail::properties_type_list<> >, sycl::_V1::detail::type_erased_cgfo_ty const&, sycl::_V1::detail::code_location const&) const (this=0x7fffffffb780, Props=..., CGF=..., CodeLoc=...)
      at /opt/intel/oneapi/compiler/2026.0/bin/compiler/../../include/sycl/queue.hpp:3700
   #15 0x00000000004041f1 in sycl::_V1::queue::submit<main::{lambda(auto:1&)#1}>(main::{lambda(auto:1&)#1}, sycl::_V1::detail::code_location const&) (this=0x7fffffffb780, CGF=..., CodeLoc=...)
      at /opt/intel/oneapi/compiler/2026.0/bin/compiler/../../include/sycl/queue.hpp:441
   #16 0x0000000000403e7c in main ()
      at Tools/ApplicationDebugger/guided_matrix_mult_Exceptions/src/2_matrix_mul_multi_offload.cpp:98

   ```

5. Examine the last frame (it may be different from the output above) using the following command:
   ```
   (gdb) frame 16
   ```
   You may need to issue this command twice before you see output similar to the following example:
   ```
   #17 0x0000000000403e7c in main ()
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

7. To fix the error, remove the extra `memcpy` from the code above (as in `3_matrix_mul.cpp`).   If this statement were actually issuing two different `memcpy` statements, you would update the code to break this up into two `submit` statements, each with a single `memcpy` .
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
