# `Guided Matrix Multiplication Exception` Sample

The `Guided Matrix Multiplication Exception` sample demonstrates a guided approach to debugging SYCL exceptions from incorrect use of the SYCL* API. It uses the Intel® oneAPI Base Toolkit (Base Kit) and several tools included in the Base Kit.

The sample code is a simple program that multiplies together two large matrices and verifies the results.

| Property              | Description
|:---                   |:---
| What you will learn   | How to use backtraces in the Intel® Distribution for GDB* to locate incorrect use of the SYCL API.
| Time to complete      | 50 minutes

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)

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
| OS                  | Ubuntu* 20.04
| Hardware            | GEN9 or newer
| Software            | Intel® oneAPI DPC++/C++ Compiler <br> Intel® Distribution for GDB*


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

The following instructions assume you have installed Intel® Distribution for GDB* and have a basic working knowledge of GDB.

To learn how setup and use Intel® Distribution for GDB*, see *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html)*.

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see *[Using the oneAPI Level Zero Interface: A Brief Introduction to the Level Zero API](https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-level-zero-interface.html?wapkw=Level%20Zero#gs.dxm4t4)*.

### Fixing the Null Pointer Version

In `1_matrix_mul_null_pointer` a null pointer is passed to a SYCL `memcpy` statement.  Unfortunately, the pointer does not tell which `memcpy` statement caused the error. To find this, we need to use a debugger (any host debugger will work; however, we will use the Intel® Distribution for GDB*).

1. Start the Intel® Distribution for GDB* debugger and run the application within the debugger.
   ```
   gdb-oneapi ./1_matrix_mul_null_pointer
   (gdb) run
   ```
2. Notice the application failure. The error is the same message seen when we ran it outside the debugger.
   ```
   terminate called after throwing an instance of 'sycl::_V1::runtime_error'
     what():  NULL pointer argument in memory copy operation. -30 (CL_INVALID_VALUE)
   Aborted (core dumped)
   ```
3. Run a `backtrace` to get a summary showing the rough location that triggered the assert.
   ```
   (gdb) backtrace 
   ```
4. Notice in the results that the exception was triggered around line 95 (frame 9):
   ```
   #0  0x00007ffff77d218b in raise () from /lib/x86_64-linux-gnu/libc.so.6
   #1  0x00007ffff77b1859 in abort () from /lib/x86_64-linux-gnu/libc.so.6
   #2  0x00007ffff7b8e951 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #3  0x00007ffff7b9a47c in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #4  0x00007ffff7b9a4e7 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #5  0x00007ffff7b9a799 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x00007ffff7ed64a1 in sycl::_V1::detail::MemoryManager::copy_usm(void const*, std::shared_ptr<sycl::_V1::detail::queue_impl>, unsigned long, void*, std::vector<_pi_event*, std::allocator<_pi_event*> >, _pi_event**) () from /home/intel/oneapi/compiler/2023.0.0/linux/lib/libsycl.so.6
   #7  0x00007ffff7f18b56 in sycl::_V1::detail::queue_impl::memcpy(std::shared_ptr<sycl::_V1::detail::queue_impl> const&, void*, void const*, unsigned long, std::vector<sycl::_V1::event, std::allocator<sycl::_V1::event> > const&) () from /home/intel/oneapi/compiler/2023.0.0/linux/lib/libsycl.so.6
   #8  0x00007ffff7fb8c09 in sycl::_V1::queue::memcpy(void*, void const*, unsigned long) () from /home/intel/oneapi/compiler/2023.0.0/linux/lib/libsycl.so.6
   #9  0x00000000004082ee in main () at /home/guided_matrix_mult/guided_matrix_mult_Exceptions/src/1_matrix_mul_null_pointer.cpp:95
   ```
5. Examine the last frame using the following (it may be different from the output above):
   ```
   (gdb) frame 9
   ```
   You should see output similar to the following example:
   ```
   #9  0x00000000004082ee in main () at /home/guided_matrix_mult/guided_matrix_mult_Exceptions/src/1_matrix_mul_null_pointer.cpp:95
   95          q.memcpy(dev_b, 0, N*P * sizeof(float));
   (gdb)
   ```

  Notice that in this case a `0` was passed as one of the pointers in the `memcpy`, which is clearly the error described in the exception. In a real application you will need to examine each of the input variables to `memcpy` using the gdb `print` command, and then trace the pointers back to where they were initialized (possibly in another source file).

### Fixing the Multiple Offload Version

In the second version, the code attempts to execute more than one offload statement in a SYCL `submit` lambda function. To find the `submit` in question, we again need to use a debugger

1. Start the Intel® Distribution for GDB* debugger and run the application within the debugger.
   ```
   gdb-oneapi ./2_matrix_mul_multi_offload
   (gdb) run
   ```
2. Notice the application failure and the location. The error is the same message seen when we ran it outside the debugger.
   ```
   terminate called after throwing an instance of 'sycl::_V1::runtime_error'
     what():  Attempt to set multiple actions for the command group. Command group must consist of a single kernel or explicit memory operation. -59 (PI_ERROR_INVALID_OPERATION)
   Thread 1 "2_matrix_mul_mu" received signal SIGABRT, Aborted.
   ```
   The assert talks about a “command group” and that only a single command group is allowed within a `submit`.  A command group is something like a `parallel_for` or a SYCL `memcpy` statement – it’s a language construct or function call that makes something happen on the device.  Only one is allowed per explicit `submit` to a queue.

3. Run a `backtrace` to get summary showing the rough location that triggered the assert.
   ```
   (gdb) backtrace
   ```

4. Notice in the results that the exception was triggered around line 98 (frame 13):
   ```
   #0  0x00007ffff77d218b in raise () from /lib/x86_64-linux-gnu/libc.so.6
   #1  0x00007ffff77b1859 in abort () from /lib/x86_64-linux-gnu/libc.so.6
   #2  0x00007ffff7b8e951 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #3  0x00007ffff7b9a47c in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #4  0x00007ffff7b9a4e7 in std::terminate() () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #5  0x00007ffff7b9a799 in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x00007ffff7f9998b in sycl::_V1::handler::memcpy(void*, void const*, unsigned long) () from /home/intel/oneapi/compiler/2023.0.0/linux/lib/libsycl.so.6
   #7  0x00000000004092b2 in main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const (this=0x7fffffffc328, h=...) at /home/guided_matrix_mult/guided_matrix_mult_Exceptions/src/2_matrix_mul_multi_offload.cpp:100
   #8  0x0000000000409145 in std::_Function_handler<void (sycl::_V1::handler&), main::{lambda(auto:1&)#1}>::_M_invoke(std::_Any_data const&, sycl::_V1::handler (__functor=..., __args=...) at /usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/bits/std_function.h:300
   #9  0x00007ffff7fba37a in sycl::_V1::detail::queue_impl::submit_impl(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /home/intel/oneapi/compiler/2023.0.0/linux/lib/libsycl.so.6
   #10 0x00007ffff7fb9945 in sycl::_V1::detail::queue_impl::submit(std::function<void (sycl::_V1::handler&)> const&, std::shared_ptr<sycl::_V1::detail::queue_impl> const&, sycl::_V1::detail::code_location const&, std::function<void (bool, bool, sycl::_V1::event&)> const*) () from /home/intel/oneapi/compiler/2023.0.0/linux/lib/libsycl.so.6
   #11 0x00007ffff7fb9905 in sycl::_V1::queue::submit_impl(std::function<void (sycl::_V1::handler&)>, sycl::_V1::detail::code_location const&) () from /home/intel/oneapi/compiler/2023.0.0/linux/lib/libsycl.so.6
   #12 0x0000000000408756 in sycl::_V1::queue::submit<main::{lambda(auto:1&)#1}>(main::{lambda(auto:1&)#1}, sycl::_V1::detail::code_location const&)(this=0x7fffffffc7a0, CGF=..., CodeLoc=...) at /home/intel/oneapi/compiler/2023.0.0/linux/bin-llvm/../include/sycl/queue.hpp:318
   #13 0x0000000000408365 in main () at /home/guided_matrix_mult/guided_matrix_mult_Exceptions/src/2_matrix_mul_multi_offload.cpp:98
   ```

5. Examine the last frame using the following (it may be different from the output above):
   ```
   (gdb) frame 13
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

   As the `assert` reported, we are trying to do two memory copies to the device within the `submit` statement, where only a single `parallel_for` or memory copy is allowed. This code needs to be recoded to use a single `submit` per `memcpy`.

7. Update the code to use a single submit.
   ```
    q.submit([&](auto &h) {
        h.memcpy(dev_c, &c_back[0], M*P * sizeof(float));
    });
   ```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).