# `Guided Matrix Multiplication Race Condition` Sample

The `Guided Matrix Multiplication Race Condition` sample demonstrates a guided approach to debugging a race condition accessing data on the host before it has been fully copied back from the device. It uses the Intel® oneAPI Base Toolkit (Base Kit) and several tools included in the Base Kit. 

The sample is a simple program that multiplies together two large matrices and verifies the results.

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)

| Property               | Description
|:---                    |:---
| What you will learn    | A way to root-cause incorrect use of the SYCL language.
| Time to complete       | 50 minutes

## Purpose

The sample in this tutorial shows how to root-cause incorrect use of the SYCL language:  accessing data on the host before it has been fully copied back from the device.

This example results in a race condition that really doesn't give any clue as to the nature of the problem.  We will show you a suite of techniques that might help you find a similar problem in more complex code.

The sample includes different versions of a simple matrix multiplication program.

| File name                           | Description
|:---                                 |:---
| `1_matrix_mul_race_condition.cpp`   |This example shows what happens when a developer tries to access data provided by the device before the copy to the host is complete.
| `2_matrix_mul.cpp`                  | A working version of the matrix multiply code that properly waits for the data to be copied back to the host.
| `3_matrix_mul.cpp`                  | A working version of the application that corrects its errors using a host accessor and a `q.wait` command in place or parenthesis.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04 <br> Windows* 10
| Hardware                | GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler <br> Intel® Distribution for GDB* <br> [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace), which is available from the [onetrace](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) GitHub repository.

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
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the *[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

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

This example shows what happens when code tries to access data provided by the device before the copy to the host is complete.

These instructions assume you have installed the Intel® Distribution for GDB* and have a basic working knowledge of GDB.

To learn how setup and use Intel® Distribution for GDB*, see the *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html)*.

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see *[Using the oneAPI Level Zero Interface: A Brief Introduction to the Level Zero API](https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-level-zero-interface.html?wapkw=Level%20Zero#gs.dxm4t4)*.

### Getting the Tracing and Profiling Tool

At an important step in this tutorial, the instructions require a utility that was not installed with the Intel® oneAPI Base Toolkit (Base Kit). 

You must download the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) code from GitHub and build the utility. The build instructions are included in the readme in the GitHub repository.

To complete the steps in the following section, you must have already built the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace). Once you have built the utility, you can invoke it before your program (similar to GBD).

### Examine the Original Code

As you might have noticed, when you attempt to run `1_matrix_mul_race_condition.cpp` the code reports bad results and then crashes when it tries to delete `c_back`. We can use the Intel® Distribution for GDB* to get a backtrace of the entire stack to understand the problem.

1. Run the Intel® Distribution for GDB*.
   ```
   gdb-oneapi 1_matrix_mul_race_condition
   ```
2. Then run the application in the debugger.
   ```
   run
   ```
3. Examine the results.
   ```
   Result of matrix multiplication using DPC++: Fail - The result is incorrect for element: [0, 0], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 1], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 2], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 3], expected: 45150, but found: 0
   Fail - The result is incorrect for element: [0, 4], expected: 45150, but found: 0
   Fail - The results mismatch!
   Abort was called at 80 line in file:
   .    ./../vpg-compute-neo/level_zero/core/source/cmdlist/cmdlist.cpp

   Thread 1 "matrix_multiply_bugged" received signal SIGABRT, Aborted.
    __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
   50      ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
   ```
   Note where the code aborted.

4. Use `backtrace` to print the current address.
   ```
   backtrace
   ```
   The output might look similar to the following:
   ```
   #0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
   #1  0x0000155554d18859 in __GI_abort () at abort.c:79
   #2  0x000015553a3da131 in ?? () from /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
   #3  0x000015553a5c9b86 in ?? () from /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
   #4  0x000015553a4ee8ac in ?? () from /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
   #5  0x000015553a4c6950 in ?? () from /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
   #6  0x0000155551ea385f in enqueueMemCopyRectHelper(_pi_command_type, _pi_queue*, void*, void*, pi_buff_rect_offset_struct*, pi_buff_rect_offset_struct*, pi_buff_rect_region_struct*, unsigned long, unsigned long, unsigned long, unsigned long, unsigned int, unsigned int, _pi_event* const*, _pi_event**, bool) ()
    from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libpi_level_zero.so
   #7  0x0000155551ea34ff in piEnqueueMemBufferReadRect () from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libpi_level_zero.so
   #8  0x00001555550fa177 in cl::sycl::detail::copyD2H(cl::sycl::detail::SYCLMemObjI*, _pi_mem*, std::shared_ptr<cl::sycl::detail::queue_impl>, unsigned int, cl::sycl::range<3>, cl::sycl::range<3>, cl::sycl::id<3>, unsigned int, char*, std::shared_ptr<cl::sycl::detail::queue_impl>, unsigned int, cl::sycl::range<3>, cl::sycl::range<3>, cl::sycl::id<3>, unsigned int, std::vector<_pi_event*, std::allocator<_pi_event*> >, _pi_event*&) () from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libsycl.so.5
   #9  0x00001555550fbc99 in cl::sycl::detail::MemoryManager::copy(cl::sycl::detail::SYCLMemObjI*, void*, std::shared_ptr<cl::sycl::detail::queue_impl>, unsigned int, cl::sycl::range<3>, cl::sycl::range<3>, cl::sycl::id<3>, unsigned int, void*, std::shared_ptr<cl::sycl::detail::queue_impl>, unsigned int, cl::sycl::range<3>, cl::sycl::range<3>, cl::sycl::id<3>, unsigned int, std::vector<_pi_event*, std::allocator<_pi_event*> >, _pi_event*&) () from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libsycl.so.5
   #10 0x0000155555154aa6 in cl::sycl::detail::MemCpyCommandHost::enqueueImp() () from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libsycl.so.5
   #11 0x000015555514c68b in cl::sycl::detail::Command::enqueue(cl::sycl::detail::EnqueueResultT&, cl::sycl::detail::BlockingT) ()
    from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libsycl.so.5
   #12 0x000015555516782b in cl::sycl::detail::Scheduler::addCopyBack(cl::sycl::detail::AccessorImplHost*) () from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libsycl.so.5
   #13 0x000015555517aed8 in cl::sycl::detail::SYCLMemObjT::updateHostMemory(void*) () from /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/lib/libsycl.so.5
   [...]
   #23 0x0000000000410689 in std::__shared_ptr<cl::sycl::detail::buffer_impl, (__gnu_cxx::_Lock_policy)2>::~__shared_ptr (this=0x7fffffffa950)
    at /usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/bits/shared_ptr_base.h:1169
   #24 0x000000000040c335 in std::shared_ptr<cl::sycl::detail::buffer_impl>::~shared_ptr (this=0x7fffffffa950)
    at /usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/bits/shared_ptr.h:103
   #25 0x000000000040bd75 in cl::sycl::buffer<float, 2, cl::sycl::detail::aligned_allocator<char>, void>::~buffer (this=0x7fffffffa950)
    at /opt/intel/oneapi/compiler/2022.0.1-prerelease/linux/bin-llvm/../include/sycl/CL/sycl/buffer.hpp:259
   #26 0x0000000000403c28 in main () at src/matrix_multiply_race_conditions.cpp:129
   ````
5. Look at the final frame. (Your frame number might differ.)
    ```
    frame 26
    ```
   You should see something similar to the following:
    ```
    #26 0x0000000000403c28 in main () at src/matrix_multiply_race_conditions.cpp:129
    129     }
    ```
6. Examine the code in that region.
    ```
    list
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

7. Exit the debugger.

8. Run the program using the [Profiling and Tracing](#getting-the-tracing-and-profiling-tool) tool.

   >**Note**: You must modify the command shown below to include the path to where you installed the `onetrace` utility.
    ```
    [path]/onetrace -c ./1_matrix_mul_race_condition
    ```
    The very last lines of the output will be something like this:
    ```
    >>>> [919469460] zeMemAllocDevice: hContext = 0x1c49480 device_desc = 0x7ffc106723c0 {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC(0x15) 0 0 0} size = 393216 alignment = 8 hDevice = 0x1c30580 pptr = 0x7ffc10672420 (ptr = 0)
    <<<< [919625755] zeMemAllocDevice [152531 ns] ptr = 0xffffd556a9d00000 -> ZE_RESULT_SUCCESS(0x0)
    >>>> [919634430] zeEventCreate: hEventPool = 0x2c2d990 desc = 0x7ffc106724d0 {ZE_STRUCTURE_TYPE_EVENT_DESC(0x11) 0 2 0 0} phEvent = 0x7ffc106724f8 (hEvent = 0)
    <<<< [919638469] zeEventCreate [1727 ns] hEvent = 0x1d37d90 -> ZE_RESULT_SUCCESS(0x0)
    >>>> [919644441] zeCommandListAppendMemoryCopyRegion: hCommandList = 0x3b62230 dstptr = 0xffffd556a9d00000 dstRegion = 0x7ffc10672578 dstPitch = 2400 dstSlicePitch = 360000 srcptr = 0x14e18dcc1010 srcRegion = 0x7ffc10672590 srcPitch = 2400 srcSlicePitch = 360000 hSignalEvent = 0 numWaitEvents = 0 phWaitEvents = 0
    <<<< [920037720] zeCommandListAppendMemoryCopyRegion [389644 ns] -> ZE_RESULT_SUCCESS(0x0)
    [...]
    <<<< [936275772] zeCommandQueueExecuteCommandLists [325053 ns] hCommandLists = 0x3b62230 -> ZE_RESULT_SUCCESS(0x0)
    Result of matrix multiplication using DPC++: Fail - The result is incorrect for element: [0, 0], expected: 45150, but found: 0
    Fail - The result is incorrect for element: [0, 1], expected: 45150, but found: 0
    Fail - The result is incorrect for element: [0, 2], expected: 45150, but found: 0
    Fail - The result is incorrect for element: [0, 3], expected: 45150, but found: 0
    Fail - The result is incorrect for element: [0, 4], expected: 45150, but found: 0
    Fail - The results mismatch!
    >>>> [1029579357] zeEventQueryStatus: hEvent = 0x28d4300
    <<<< [1029596743] zeEventQueryStatus [4684 ns] -> ZE_RESULT_NOT_READY(0x1)
    >>>> [1029610715] zeEventCreate: hEventPool = 0x2c2d990 desc = 0x7ffc10672720 {ZE_STRUCTURE_TYPE_EVENT_DESC(0x11) 0 4 0 0} phEvent = 0x7ffc10672748 (hEvent = 0)
    <<<< [1029621736] zeEventCreate [6042 ns] hEvent = 0x1b8ca50 -> ZE_RESULT_SUCCESS(0x0)
    >>>> [1029628259] zeCommandListAppendWaitOnEvents: hCommandList = 0x3c98b40 numEvents = 1 phEvents = 0x28c9730 (hEvents = 0x1d63150)
    <<<< [1029634854] zeCommandListAppendWaitOnEvents [3692 ns] hEvents = 0x1d63150 -> ZE_RESULT_SUCCESS(0x0)
    >>>> [1029642108] zeCommandListAppendMemoryCopyRegion: hCommandList = 0x3c98b40 dstptr = 0x14e18dcc1010 dstRegion = 0x7ffc106727c8 dstPitch = 2400 dstSlicePitch = 360000 srcptr = 0xffffd556a9d00000 srcRegion = 0x7ffc106727e0 srcPitch = 2400 srcSlicePitch = 360000 hSignalEvent = 0 numWaitEvents = 0 phWaitEvents = 0
    Abort was called at 80 line in file:
    ../../neo/level_zero/core/source/cmdlist/cmdlist.cpp
    ```

### Interpret the Results

The first clue here is that the debugger reports the code crashes **when the program exits** after it has completed checking the results and finding them bad. That behavior is worrying.

Next, looking at the crash in the debugger, there are a couple of odd things that stand out.   Look at stack frames 6-13.  These frames are all about updating the host memory from the device, while frame 26 shows we are already at the end of the program and have started cleaning up the buffers (frame 25).  The only variable expected back from the device is `c_back`.  But the developer has already deleted `c_back` in line 126, so the *data the buffer being copied into `c_back` and simultaneously deleted is no longer valid*.

We see something like this in the `onetrace` output above.   The kernel is executed, the results are immediately checked, we create and wait on some events, and then the last thing we try to do before crashing is to copy some memory from the device memory (`srcptr = 0xffffd556a9d00000`) to a host pointer (`dstptr = 0x14e18dcc1010`) that previously was used to initialize this same device memory (around line 101).   Since `c_buf` is the only accessor that is defined as writeable, it again is a likely suspect.

But what if the developer didn't delete `c_back`, and let program termination clean it up?  Try it!  Unfortunately, in that case your program complains about bad results, but it exits cleanly (shutdown will wait for the GPU to copy memory back to the host buffer before it kills the buffer).

Is the behavior different if you run it on OpenCL or Level 0?  The default is to use the Level Zero run time, but we can explicitly force the use of either Level Zero or OpenCL, which can be helpful when troubleshooting.

```
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./1_matrix_mul_race_condition
ONEAPI_DEVICE_SELECTOR=opencl:gpu ./1_matrix_mul_race_condition
```
Unfortunately not; pretty much the same thing happens.

Similarly, we specify targeting the CPU, which sometimes can avoid problems in your code that are specific to offloading to the GPU.
```
ONEAPI_DEVICE_SELECTOR=*:cpu ./1_matrix_mul_race_condition
```
This also has problems, but if you run this in the debugger you will see lots of threads all running in the third `q.submit` kernel, but no thread running `main`.   This is because these threads have been abandoned when `main` deleted `c_back` and exited!

So in conclusion, it looks like the third kernel is still executing and/or its results are still being copied back to the host as the program is terminating.  Which explains the incorrect results (they aren't available on the host yet) and the crash (results from the card are being copied to memory that has been deallocated).  All these point to some sort of synchronization issue or race condition between the host and device.

### Understand the Problem

Because we are using SYCL buffers, even though the `q.submit` statements that populate `a_buf` and `b_buf` execute asynchronously, the third `q.submit` statement does not execute until those first two submits are complete because the SYCL runtime realizes that the third `q.submit` depends on the `a_buf` and `b_buf` buffers, which are being used in the first two kernels.   Once the first two kernels complete, the third `q.submit` kernel starts executing because its input data are ready. The SYCL runtime them immediately returns control to the host and we proceed to the code which verifies the result - **while the third `q.submit` keeps running**.

There are three errors in this code:

1. As just noted, we did not wait until the third `q.submit` kernel completed before accessing the data in `c_back`. This could either be done using parenthesis to enforce scope, and thus order of operations, or by adding a `q.wait()` call just before the call to `VerifyResult`.

2. We  should be using a host accessor pointing to buffer `c_buf` to access its contents, which would also indicate that we need to wait for the third `q.submit kernel` to complete **and** for the *data to be copied back to the host* before accessing the data in `c_back`.

3. For buffers initialized with a pointer to host memory (like `c_buf`), the developer agrees to not touch the host pointer again until the buffer is destroyed, so deleting the host memory before the buffer is destroyed is illegal (the call to `delete[] c_back;`). The buffer cannot detect that the memory was deallocated.

### Fix the Code

Let's fix one of these bugs at a time.

To address (a), go back into the program and add a `q.wait()` before the call to `VerifyResult`:

```
q.wait();

int result;
cout << "Result of matrix multiplication using DPC++: ";
result = VerifyResult(c_back);
delete[] c_back;
```

To address (b), we need to signal the SYCL runtime that the results of the kernel are needed back on the host for our `VerifyResult` call. We do this using a host accessor.

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
The result should look like `3_matrix_mul.cpp`.  With these changes we did the following:
1.  We created a host accessor to pull the values of out `c_buf` on the host, forcing the data to be transferred from the device to the host before the first access (one of the race conditions in this code).
2.  We are waiting for the third `q.submit` kernel to complete before asking for the values in `c_buf`, fixing the other race condition.
3.  We are no longer deleting `c_back` before the buffer that makes use of it (`c_buf`) is destroyed on program exit.
4.  We changed `VerifyResult` to pass down the host accessor, which we are able to read just like the original `c_back` array.

### Examine the Problem from a Different Perspective

Compare `1_matrix_mul_race_condition.cpp` and `2_matrix_mul.cpp` source files.

Note that the source files differ by **two characters** (parentheses) only.

```
54d53
<   {
123d121
<   }
```
This is a more detailed region of the code.
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

Realizing that in the real world we may not have such simple source at hand, let's look a little more closely.  The brackets in question that `2_matrix_mul.cpp` contains start just before the SYCL queue is defined, and end before we look at the results in `c_back`.

As a result, because the device queue `q` exists only in the scope of those brackets, host execution waits at the brackets until the third kernel completes and the updates made to the write-only accessor `c` pointing to `c_buf` get copied from the device and into `c_back`.

Note that `2_matrix_mul.cpp` still has a bug.  It is an example of problem (b) above - it's not using a  host accessor to access the data in `c_back`, which is still being managed by buffer `c_buf`.   We just got lucky.

This points out a potential trap in the training documentation you may have read while learning SYCL.   You could easily get the impression that if you use the buffer-accessor mechanism, synchronization will be taken care of for you.  The use of parenthesis may be mentioned in passing with little explanation.   Even though the documentation may say "the { } block ensures all SYCL work has concluded," this is not stressed.

This is the trap of the buffer-accessor mechanism - you may assume that the automatic synchronization mechanism is smarter than it really is.  In `1_matrix_mul_race_condition.cpp`, the SYCL runtime does not realize that we cannot call `VerifyResult` with the `c_back` array until the third `q.submit` kernel completes and the data are copied back to the host - it assumes you know what you are doing. 

>**Note**: You will find more the proper use of buffers and accessors in the *Buffer Accessor Mode* section of the *[oneAPI GPU Optimization Guide Developer Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/memory/buffer-accessors.html)*.

### Generalized Approach

As we saw, `1_matrix_mul_race_condition.cpp` suffered from multiple race conditions and directly manipulating data contained in a buffer without informing the buffer (this was an extreme case - we deleting the underlying data managed by the buffer).

But this gives us some hints on how to find these types of problems in any sort of code.

1.  The kernel is producing correct results, but they are not getting to the host when the host tries to access them.
2.  The host crashes when your host code calls a buffer destructor (frame 25 above).
3.  The host crashes when the SYCL runtime attempts to copy device data to a buffer on the host (frames 6-13 above).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).