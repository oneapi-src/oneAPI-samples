# `Guided Matrix Multiplication Invalid Contexts` Sample

The `Guided Matrix Multiplication Invalid Contexts` sample demonstrates how to use the Intel® oneAPI Base Toolkit (Base Kit) and several tools found in it to triage incorrect use of the SYCL language.

The sample is simple program that multiplies together two large matrices and verifies the results.

| Property              | Description
|:---                   |:---
| What you will learn   | A method to determine the root cause of incorrect use of queues with different contexts.
| Time to complete      | 50 minutes

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The sample in this tutorial shows how to debug incorrect use of variables that are owned by different queues that have different contexts.

This type of error can be hard to detect and determine the root cause in a large body of code where queues and memory are passed between functions. The lack of tools that tell you what is wrong combined with the fact that the default Level Zero driver does not notice there is a problem (only the OpenCL™ driver and CPU-side runtimes report the issue) make this issue particularly painful since code that runs on a single device can fail to run on two devices with no indication why.

The sample includes different versions of a simple matrix multiplication program.

| File                                 | Description
|:---                                  |:---
| `1_matrix_mul_invalid_contexts.cpp`  | This example shows what happens when a developer mixes up queues owned by different contexts.
| `2_matrix_mul.cpp`                   | A working version of the matrix multiply code that uses the same queue for all memory operations.

## Prerequisites

| Optimized for       | Description
|:---                 |:---
| OS                  | Ubuntu* 20.04
| Hardware            | GEN9 or newer
| Software            | Intel® oneAPI DPC++/C++ Compiler <br> Intel® Distribution for GDB* <br> [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace), which is available from the [onetrace](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) GitHub repository.

## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* queues and devices
- Explicit memory operations using Unified Shared Memory (USM)  
- SYCL* kernels (including parallel_for function and explicit memory copies)
- SYCL* queues

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build and Run the `Guided Matrix Multiplication Invalid Contexts` Programs 

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

   For the mixed queue version only, enter the following:
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

This guided example demonstrates what might happen when a developer mixes up queues owned by different contexts. 

### Getting the Tracing and Profiling Tool

At an important step in this tutorial, the instructions require a utility that was not installed with the Intel® oneAPI Base Toolkit (Base Kit).

You must download the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace) code from GitHub and build the utility. The build instructions are included in the readme in the GitHub repository.

### Check the Programs

1. Notice that both versions of the application run to completion and report correct results.

   SYCL applications use the Level Zero runtime by default with an Intel GPU. If you use OpenCL™ software to run `1_matrix_mul_invalid_contexts`, the program with a bug in it will crash before it can report results.

2. Check the results on a **GPU** with OpenCL.
   ```
   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./1_matrix_mul_invalid_contexts
   ```
   The output might look similar to the following:
   ```
   Initializing
   Computing
   Device: Intel(R) Graphics [0x020a]
   Device compute units: 960
   Device max work item size: 1024, 1024, 1024
   Device max work group size: 1024
   Problem size: c(150,600) = a(150,300) * b(300,600)
   terminate called after throwing an instance of 'cl::sycl::runtime_error'
     what():  Native API failed. Native API returns: -5 (CL_OUT_OF_RESOURCES) -5 (CL_OUT_OF_RESOURCES)
   Aborted (core dumped)
   ```
3. Check the results on the **CPU** using OpenCL. You should see similar problems.
   ```
   ONEAPI_DEVICE_SELECTOR=opencl:cpu ./1_matrix_mul_invalid_contexts
   ```
   The output might look like the following:
   ```
   Initializing
   Computing
   Device: Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz
   Device compute units: 144
   Device max work item size: 8192, 8192, 8192
   Device max work group size: 8192
   terminate called after throwing an instance of 'cl::sycl::runtime_error'
     what():  No device of requested type available. Please check https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html -1 (CL_DEVICE_NOT_FOUND)
   Aborted (core dumped)
   ```

Note the change in results. In the next section, let us examine what went wrong.

### Use the Debugger to Find the Issue

In this section, you will use the Intel® Distribution for GDB* to determine what might be wrong.

1. Start the debugger using OpenCL™ on the **GPU**.
   ```
   ONEAPI_DEVICE_SELECTOR=opencl:gpu  gdb-oneapi ./1_matrix_mul_invalid_contexts
   ```
2. You should get the prompt `(gdb)`.

3. From the debugger, run the program.
   ```
   (gdb) run
   ```
   This will launch the application and provide some indication of where the code failed.

   ```
   Starting program: .../1_matrix_mul_invalid_contexts 
   :
   [Thread debugging using libthread_db enabled]
   Using host libthread_db library "/usr/lib/x86_64-linux-gnu/libthread_db.so.1".
   Initializing
   Computing
   Device: Intel(R) Graphics [0x020a]
   Device compute units: 960
   Device max work item size: 1024, 1024, 1024
   Device max work group size: 1024
   Problem size: c(150,600) = a(150,300) * b(300,600)
   [New Thread 0x15553b1c8700 (LWP 47514)]
   terminate called after throwing an instance of 'cl::sycl::runtime_error'
     what():  Native API failed. Native API returns: -5 (CL_OUT_OF_RESOURCES) -5 (CL_OUT_OF_RESOURCES)

   Thread 1 "matrix_multiply_bugg" received signal SIGABRT, Aborted.
   __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
   50      ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
   (gdb)
   ```

4. Prompt for a call stack to inspect the results.
   ```
   (gdb) where
   ```
   The output can be extensive, and might look similar to the following:

   ```
   #0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
   #1  0x0000155554ce4859 in __GI_abort () at abort.c:79
   #2  0x00001555553da911 in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
   #3  0x00001555553e638c in ?? () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
   #4  0x00001555553e63f7 in std::terminate() () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
   #5  0x00001555553e637f in std::rethrow_exception(std::__exception_ptr::exception_ptr) () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6
   #6  0x0000155555153fe4 in cl::sycl::detail::Scheduler::addCG(std::unique_ptr<cl::sycl::detail::CG, std::default_delete<cl::sycl::detail::CG> >, std::shared_ptr<cl::sycl::detail::queue_impl>) () from /opt/intel/oneapi_customer/compiler/2022.1.0/linux/lib/libsycl.so.5
   #7  0x000015555518ef30 in cl::sycl::handler::finalize() () from /opt/intel/oneapi_customer/compiler/2022.1.0/linux/lib/libsycl.so.5
   #8  0x00001555551bc3ea in cl::sycl::detail::queue_impl::finalizeHandler(cl::sycl::handler&, cl::sycl::detail::CG::CGTYPE const&, cl::sycl::event&) ()
   from /opt/intel/oneapi_customer/compiler/2022.1.0/linux/lib/libsycl.so.5
   #9  0x00001555551bc13b in cl::sycl::detail::queue_impl::submit_impl(std::function<void (cl::sycl::handler&)> const&, std::shared_ptr<cl::sycl::detail::queue_impl> const&, std::shared_ptr<cl::sycl::detail::queue_impl> const&, std::shared_ptr<cl::sycl::detail::queue_impl> const&, cl::sycl::detail::code_location const&, std::function<void (bool, bool, cl::sycl::event&)> const*) () from /opt/intel/oneapi_customer/compiler/2022.1.0/linux/lib/libsycl.so.5
   #10 0x00001555551bb744 in cl::sycl::detail::queue_impl::submit(std::function<void (cl::sycl::handler&)> const&, std::shared_ptr<cl::sycl::detail::queue_impl> const&, cl::sycl::detail::code_location const&, std::function<void (bool, bool, cl::sycl::event&)> const*) () from /opt/intel/oneapi_customer/compiler/2022.1.0/linux/lib/libsycl.so.5
   #11 0x00001555551bb715 in cl::sycl::queue::submit_impl(std::function<void (cl::sycl::handler&)>, cl::sycl::detail::code_location const&) ()
   from /opt/intel/oneapi_customer/compiler/2022.1.0/linux/lib/libsycl.so.5
   #12 0x0000000000404536 in cl::sycl::queue::submit<main::{lambda(auto:1&)#1}>(main::{lambda(auto:1&)#1}, cl::sycl::detail::code_location const&) (this=0x7fffffffa750, CGF=..., 
    CodeLoc=...) at /opt/intel/oneapi_customer/compiler/2022.1.0/linux/bin-llvm/../include/sycl/CL/sycl/queue.hpp:275
   #13 0x000000000040408a in main () at src/1_matrix_mul_invalid_contexts.cpp:101
   (gdb) 
   ```

5. Note that the last frame number in the call stack (your last frame may vary from the example above).
6. Switch the debugger focus to that frame.
   ```
   (gdb) frame 13
   ```
   Your output will be similar to the following:
   ```
   #13 0x000000000040408a in main () at src/1_matrix_mul_invalid_contexts.cpp:101
   101         q.submit([&](auto &h) {
   (gdb) 
   ```
7. Examine the source code in that region.
   ```
   (gdb) list
   ```
   You should see the code around the line reporting the problem.
   
   ```
   96
   97          // Submit command group to queue to initialize matrix b
   98          q.memcpy(dev_b, &b_back[0], N*P * sizeof(float));
   99
   100         // Submit command group to queue to initialize matrix c
   101         q.submit([&](auto &h) {
   102             h.memcpy(dev_c, &c_back[0], M*P * sizeof(float));
   103         });
   104
   105         q.wait();
   (gdb) 
   ```

   As you can see, there is something wrong in line 101.  Unfortunately, the generic `CL_OUT_OF_RESOURCES` we saw when it crashed doesn't really mean anything - it just tells us there is a problem.

   Fortunately, in this case the two variables, `dev_c` and `c_back`, are allocated only a few lines above line 101. In real code this might have happened in another source file or library, so hunting down this issue is going to be much harder.

   Look at the source, and note that `dev_c` is defined as:
   ```
   float * dev_c = sycl::malloc_device<float>(M*P, q2);
   ```
   and `c_back` is defined as follows as local memory
   ```
   float(*c_back)[P] = new float[M][P];
   ```

8. Look at line 101, and notice the discrepancy.
   ```
   q.submit([&](auto &h) {
   ```
   Variable `dev_c` was allocated on queue `q2` while the submit statement is being done on queue `q`.

### Identify the Problem without Code Inspection

You must have already built the [Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/onetrace). Once you have built the utility, you can invoke it before your program (similar to GBD).

One of the things that the Tracing and Profiling utility can help us identify is printing every low-level API call made to OpenCL™ or Level Zero. This is the features that we will use to attempt to match the source to the events.

1. Let's look at the output from using OpenCL again since the program stopped when it hit a failure previously.<br> Include `-c` when invoking `onetrace` to enable call logging of API calls.

   >**Note**: You must modify the command shown below to include the path to where you installed the `onetrace` utility.

   ```
   ONEAPI_DEVICE_SELECTOR=opencl:gpu [path]/onetrace -c ./1_matrix_mul_invalid_contexts
   ```

   The `onetrace` utility outputs extensive results. A few key excerpts with areas of interest are shown below.

   ```
    `>>>>` [10826025] clCreateCommandQueueWithProperties: <mark>*context = 0x15cb110*</mark> device = 0x111daa0 properties = 0x7ffd83a771b0 errcodeRet = 0x7ffd83a771a4<br>
    `<<<<` [10839130] clCreateCommandQueueWithProperties [9913 ns] <mark>*result = 0x15cd980*</mark> -> CL_SUCCESS (0)<br>
   [...]<br>
    `>>>>` [16672600] clDeviceMemAllocINTEL: <mark>**context = 0x15cb790**</mark> device = 0x111daa0 properties = 0x7ffd83a77620 size = 360000 alignment = 4 errcode_ret = 0x7ffd83a7721c<br>
    `<<<<` [17003160] clDeviceMemAllocINTEL [330560 ns] <mark>**result = 0xffffd556aa680000**</mark> -> CL_SUCCESS (0)<br>
   [...]<br>
    `>>>>` [25836849] clEnqueueMemcpyINTEL: <mark>*command_queue = 0x15cd980*</mark> blocking = 0 <mark>**dst_ptr = 0xffffd556aa680000**</mark> src_ptr = 0x153db9e39010 size = 360000 num_events_in_wait_list = 0 event_wait_list = 0 event = 0x15dc9c8<br>
    `<<<<` [25918680] clEnqueueMemcpyINTEL [81831 ns] -> CL_OUT_OF_RESOURCES (-5)
   ```

   Let's work backwards from the error, starting with `clEnqueueMemcpyINTEL`.

   This function uses `command_queue = 0x15cd980` and copies `src_ptr` into device memory `dst_ptr = 0xffffd556aa680000`. Working back up the stack, you can see we allocated the device memory with the address `0xffffd556aa680000` using device context `0x15cb790`.  However, the command queue being used in the `clEnqueueMemcpyINTEL` was created using the device context `0x15cb110`, which is different from the device context used to allocate the destination memory. The generic error we get is the OpenCL indication stating that this is illegal.

   This **is** legal if both queues point to the same device context; however, in this example `q2` is actually defined pointing to another device context. You might do this in actual code if you have multiple offload compute devices you are targeting. This code is sending work and data to each device for processing. It is easy to send the wrong pointer to the wrong queue accidentally in complex code.

2. Look at the output from Level Zero, and see if we could have detected the issue.
   ```
   ONEAPI_DEVICE_SELECTOR=level_zero:gpu [path]onetrace -c ./1_matrix_mul_invalid_contexts
   ```
   Your output might be similar to the following:
   ```
    `>>>>` [17744815] zeMemAllocDevice: <mark>**hContext = 0x1fbb640**</mark> device_desc = 0x7ffde3fab480 {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC(0x15) 0 0 0} size = 393216 alignment = 8 hDevice = 0x183fae0 pptr = 0x7ffde3fab4f8 (ptr = 0)  
    `<<<<` [17910069] zeMemAllocDevice [163062 ns] <mark>**ptr = 0xffffd556aa660000**</mark> -> ZE_RESULT_SUCCESS(0x0)  
   [...]  
    `>>>>` [18915736] zeCommandListCreate: <mark>*hContext = 0x1f9f290*</mark> hDevice = 0x183fae0 desc = 0x7ffde3fab480 {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC(0xf) 0 1 0} phCommandList = 0x7ffde3fab470 (hCommandList = 0x7ffde3fab480)  
    `<<<<` [18937865] zeCommandListCreate [20335 ns] <mark>*hCommandList = 0x1fccbd0*</mark> -> ZE_RESULT_SUCCESS(0x0)  
   [...]  
    `>>>>` [19186206] zeCommandListAppendMemoryCopy: <mark>*hCommandList = 0x1fccbd0*</mark> <mark>**dstptr = 0xffffd556aa660000**</mark> srcptr = 0x15274c1d0010 size = 360000 hSignalEvent = 0x1fce9a0 numWaitEvents = 0 phWaitEvents = 0
    `<<<<` [19211401] zeCommandListAppendMemoryCopy [23937 ns] -> ZE_RESULT_SUCCESS(0x0)
   ```
   Once again, if we look closely, we can see that we are using two different device contexts, which is potentially illegal.  On a single-card, single-tile system this may be legal, but if you are using different device contexts to access different tiles, or different offload devices, the program will crash in those cases but not in the single-card, single-tile system. This could make debugging quite challenging.

   The `onetrace` utility can be very helpful detecting these sorts of issues at a low level.  It has other useful abilities, like telling you how much runtime each OpenCL or Level Zero call consumed, how long your kernels ran on the device, how long memory transfers took, and it can even create a timeline showing what happened when using the Chrome(tm) tracing browser tool.

### Fix the Problem

To fix this problem, you must change the `malloc_device` allocating `dev_c` to use the same queue (and thus same device context) as the first two device allocations or create queue `q` to use the same underlying device context.

If you really need to operate over multiple devices, this example will need to be entirely re-written, which is beyond the scope of this tutorial.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).