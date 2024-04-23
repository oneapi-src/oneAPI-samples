# `jacobi` Sample

The 'jacobi' code sample is a small SYCL*-compliant sample for exercising application
debugging using Intel&reg; Distribution for GDB*. It is highly recommended that
you go through this sample *after* you familiarize yourself with the
[array-transform](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/ApplicationDebugger/array-transform)
sample and the
[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html)
for the debugger.

This sample contains two versions of the same program: `jacobi-bugged` and
`jacobi-fixed`. The latter works correctly, but in the former several bugs are
injected. You can try to find and fix them using the debugger.  The debug steps
follow a common strategy that attempts to resolve bugs first on the CPU, 
then focus on possibly more difficult GPU-oriented bugs.

| Area                | Description
|:---                 |:---
| What you will learn | Find existing bugs in a program using the debugger
| Time to complete    | 1 hour for CPU or FPGA emulator; 2 hours for GPU

## Prerequisites

| Optimized for                                    | Description
|:---                                              |:---
| OS                                               | Linux* Ubuntu* 20.04 to 22.04 <br> CentOS* 8 <br> Fedora* 30 <br> SLES 15
| Hardware to debug offloaded <br> kernels on GPUs | Intel® Arc(tm) <br> Intel® Data Center GPU Flex Series
| Software                                         | Intel&reg; oneAPI DPC++/C++ Compiler

> **Note** although the sample can be run on all supported by Intel® oneAPI
> Base Toolkit platforms, the GPU debugger can debug only kernels offloaded
> onto devices specified at “Hardware to debug offloaded kernels on GPUs”
> while running with the L0 backend.  When the GPU device is different from
> the listed above, e.g., an integrated graphics device, breakpoints inside
> the kernel won't be hit.  In such case, try to switch the offload to a CPU
> device by using ONEAPI_DEVICE_SELECTOR environment variable.

We recommend to first make sure that the program you intend to debug is running
correctly on CPU and only after that switch the offload to GPU.

> *Note**: although the sample can be built and run on Windows* 10 and 11 too,
> we focus on demonstrating how the GDB* interface can be used
> to examine the program on Linux* OS.

## Purpose

The `jacobi` sample is a SYCL application that solves a hardcoded linear system
of equations `Ax=b` using the Jacobi iteration. The matrix `A` is an `n x n` sparse
matrix with diagonals `[1 1 5 1 1]`. Vectors `b` and `x` are `n x 1` vectors.
Vector `b` is set in the way that all components of the solution vector `x` are 1.

```
            A                 x   =   b

[5 1 1 0 0     ... 0 0 0 0]  [1]     [7]
[1 5 1 1 0 0 0 ... 0 0 0 0]  [1]     [8]
[1 1 5 1 1 0 0 ... 0 0 0 0]  [1]     [9]
[0 1 1 5 1 1 0 0 ... 0 0 0]  [1]     [9]
[0 0 1 1 5 1 1 0 0 ... 0 0]  [1]  =  [9]
[...]                       [...]   [...]
[0 0 0 0 ... 0 1 1 5 1 1 0]  [1]     [9]
[0 0 0 0 ... 0 0 1 1 5 1 1]  [1]     [8]
[0 0 0 0 ... 0 0 0 0 1 1 5]  [1]     [7]
```

The program `jacobi-fixed` computes vector `x` correctly. That is, all
components of the resulting vector are close to 1.0. However, `jacobi-bugged`
returns an incorrect result. There are three bugs, two of them you can fix on CPU,
the third one becomes visible on GPU. We recommend to follow the order of bugs:
first fix Bug 1 and Bug 2 on CPU, then switch to GPU and fix Bug 3.

The sample is intended for exercising the debugger, not for performance
benchmarking.

The debugger supports debugging kernels that run on the CPU, GPU, or accelerator
devices.  Use the ONEAPI_DEVICE_SELECTOR environment variable
to select device.  The default device is Level Zero GPU device, if available.
For more details on possible values of this variable see [Environment Variables](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector).

For an overview of the Jacobi method, please refer to the Wikipedia article 
on the [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method).

For an overview of the debugger, please refer to 
[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html).

## Key Implementation Details

Jacobi method is an iterative solver for a system of linear equations. The system
must be strictly diagonally dominant to ensure that the method converges
to the solution. In our case, the matrix is hardcoded into the kernel from
`compute_x_k1`.

All computations happen inside a while-loop. There are two exit criteria
from the loop:
* *success* if the relative error falls below the desired tolerance,
  the algorithm converged.
* *fail* if we exceed the maximum number of iterations, the algorithm
  did not converge.

There are 3 files in the solution.

| File          | Description
|:---           |:---
| jacobi.cpp    | Contains the `main` function, no bugs: <br> * initialize the problem <br> * setup SYCL boiler plate <br> call the `iterate` function <br> validate the result
| bugged.cpp    | Contains bugged versions of `iterate`, `compute_x_k1`, and `prepare_for_next_iteration` functions. <br> This file is included only into `jacobi-bugged` program.
| fixed.cpp     | Contains fixed versions of `iterate`, `compute_x_k1`, and `prepare_for_next_iteration` functions. <br> This file is included only into `jacobi-fixed` program.

The iteration loop is located at `iterate` function.
Each iteration has two parts: `compute_x_k1` and `prepare_for_next_iteration`.
All intended bugs are located in the `bugged.cpp` file.

### Update x_k1 vector (`compute_x_k1`)

Here we compute the resulting vector of this iteration `x^{k+1}`.

Each iteration of the Jacobi method performs the following update for
the resulting vector:

```
x^{k+1} = D^{-1}(b - (A - D)x^k),
```

where `n x n` matrix `D` is a diagonal component of the matrix `A`.
Vector `x^k` is the result of the previous iteration (or an initial guess
at the first iteration). In the sample, this computation is offloaded
to the device. In the code, `x^{k+1}` corresponds to the variable `x_k1`,
and `x^k` corresponds to `x_k`.

### Relative error computation

At each iteration we compute the relative error as:

```
                  ||x_k - x_k1||_1
relative_error = ------------------,
                     ||x_k1||_1
```

where absolute error `||x_k - x_k1||_1` is L1 norm of vector `x_k - x_k1` and
`||x_k1||_1` is L1 norm of vector `x_k1`.

L1 norm of a vector is the sum of absolute values of its elements.

To compute L1 norms of `||x_k - x_k1||_1` and `||x_k1||_1` we use
sum reduction algorithm. This happens in `prepare_for_next_iteration`.
The computation of the final relative error happens in `iterate`:
we divide the absolute error `abs_error` by the L1 norm of x_k1 (`l1_norm_x_k1`).

### Prepare for the next iteration

First, we perform sum-reductions over the vectors `x_k1 - x_k` and `x_k1`
to compute L1 norms of `x_k - x_k1` and `x_k1` respectively.

Second, we update `x_k` with the new value from `x_k1`.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure 
the oneAPI toolkits using environment variables. Set up your CLI environment 
by sourcing the `setvars` script every time you open a new terminal window. 
This practice ensures that your compiler, libraries, and tools are ready
for development.

> **Note**: The `setvars` script is located in the root of your
> oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

## Build the `jacobi-bugged` and `jacobi-fixed` programs

Preliminary setup steps are needed for the debugger to function.
Please see the setup instructions in the Get Started Guide
[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/)

### Using Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### Using Intel&reg; DevCloud

If running a sample in the Intel&reg; DevCloud, remember that you must specify the
compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.

We recommend running the `jacobi` sample on a node with GPU. In order to have an
interactive debugging session, we recommend using the interactive mode. To get
the setting, after connecting to the login node, type the following command:

```
$ qsub -I -l nodes=1:gpu:ppn=2
```

Within the interactive session on the GPU node, build and run the sample.
For more information, see the Intel® oneAPI Base Toolkit Get Started Guide
(https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### Build on a Linux* System

Build the project using the following `cmake` commands.

    ```
    $ cd jacobi
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```

    This builds both `jacobi-bugged` and `jacobi-fixed` versions of the program.
    > Note: The cmake configuration enforces the `Debug` build type.

### Setup the Debugger

Preliminary setup steps are needed for the debugger to function.
Please see the setup instructions in the Get Started Guide 
[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/)

### Run the buggy program

    ```
    $ ./jacobi-bugged
    ```
    > Note: to specify a device type to offload the kernel, use
    > the `ONEAPI_DEVICE_SELECTOR` environment variable.
    > E.g.  to restrict the offload only to CPU devices use:
    ```
    $ ONEAPI_DEVICE_SELECTOR=*:cpu ./jacobi-bugged
    ```

3.  Start a debugging session on a CPU device:

    ```
    $ ONEAPI_DEVICE_SELECTOR=*:cpu gdb-oneapi jacobi-bugged
    ```

4.  Clean the program (optional):

    ```
    $ make clean
    ```


For instructions about starting and using the debugger on Linux* OS Host,
please see
- [Tutorial: Debug a SYCL* Application on a CPU](https://www.intel.com/content/www/us/en/develop/documentation/debugging-dpcpp-linux/top/debug-a-sycl-application-on-a-cpu.html)
- [Tutorial: Debug a SYCL* Application on a GPU](https://www.intel.com/content/www/us/en/develop/documentation/debugging-dpcpp-linux/top/debug-a-sycl-application-on-a-gpu.html)

## Guided Debugging

The below instructions provide step by step instructions for locating and
resolving the three bugs in the `jacobi` sample, as well as basic usage of
the debbuger. 

### Recommended Commands

For checking variables values you can use: `print`, `printf`, `display`,
`info locals`.

#### `commands` command

To define specific actions when a BP is hit, use `commands <breakpoint number>`,
e.g.

```
(gdb) commands 1
Type commands for breakpoint(s) 1, one per line.
End with a line saying just "end".
>silent
>print var1
>continue
>end
```

The above sets actions for breakpoint 1. At each hit of the breakpoint,
the variable `var1` will be printed and then the debugger will continue until
the next breakpoint hit.  Here we start the command list with `silent` -- it
helps to suppress GDB output about hitting the BP.

On GPU all threads execute SIMD. To apply the command to every SIMD lane
that hit the BP, add the `/a` modifier. In the following we print the local
variable `gid` for every SIMD lane that hits the breakpoint at `x_k1_kernel`:

```
(gdb) break compute_x_k1_kernel
Breakpoint 1 at 0x4048e8: file bugged.cpp, line 18.
(gdb) commands /a
Type commands for breakpoint(s) 1, one per line.
Commands will be applied to all hit SIMD lanes.
End with a line saying just "end".
>print index
>end
(gdb) running
<... GDB output omitted...>

[Switching to Thread 1.153 lane 0]

Thread 2.153 hit Breakpoint 1, with SIMD lanes [0-7], compute_x_k1_kernel (index=sycl::_V1::id<1> = {...}, b=0xffffb557aa530000, x_k=0xffffb557aa530200, x_k1=0xffffb557aa530400) at bugged.cpp:18
18        int i = index[0];
$1 = sycl::_V1::id<1> = {56}
$2 = sycl::_V1::id<1> = {57}
$3 = sycl::_V1::id<1> = {58}
$4 = sycl::_V1::id<1> = {59}
$5 = sycl::_V1::id<1> = {60}
$6 = sycl::_V1::id<1> = {61}
$7 = sycl::_V1::id<1> = {62}
$8 = sycl::_V1::id<1> = {63}
```

#### `thread apply`

You can apply a command to specified threads and SIMD lanes with `thread apply`.
The following command prints `index` for each active SIMD lane of the current
thread:

```
(gdb) thread apply :* -q print index
$9 = sycl::_V1::id<1> = {56}
$10 = sycl::_V1::id<1> = {57}
$11 = sycl::_V1::id<1> = {58}
$12 = sycl::_V1::id<1> = {59}
$13 = sycl::_V1::id<1> = {60}
$14 = sycl::_V1::id<1> = {61}
$15 = sycl::_V1::id<1> = {62}
$16 = sycl::_V1::id<1> = {63}
```

The `-q` flag suppresses the additional output such as for which thread and lane
the command was executed.

#### User-defined Convenience Variables

You can define a convenience variable and use it for computations from
the debugger.

In the following we define a variable `$temp` and set its initial value to `1`.
Then we increment its value.
```
(gdb) set $temp=1
(gdb) print $temp
$1 = 1
(gdb) print ++$temp
$2 = 2
```

#### Stepping through the program

If you want to step through the program, use: `step`, `next`. Ensure that
scheduler locking is set to `step` or `on` (otherwise, you will be switched
to a different thread while stepping):

```
(gdb) set scheduler-locking step
```

### Debugging `jacobi-bugged`

Again, to try to isolate our bugs, we'll focus first on the CPU.  This is 
a common strategy.  You can specify the device for offloading the kernels
using `ONEAPI_DEVICE_SELECTOR` variable, for example: 

```
$ ONEAPI_DEVICE_SELECTOR=*:cpu ./myApplication`. 
```

The above limits the kernel offloading only to CPU devices.

#### 1. Run the application on CPU

When run, the original `jacobi-bugged` program shows the first bug:

```
$ ONEAPI_DEVICE_SELECTOR=*:cpu ./jacobi-bugged
[SYCL] Using device: [Intel(R) Core(TM) i7-7567U processor] from [Intel(R) OpenCL]
Iteration 0, relative error = 2.71116
Iteration 20, relative error = 1.70922
Iteration 40, relative error = 1.10961
Iteration 60, relative error = 0.818992
Iteration 80, relative error = 0.648988

fail; Bug 1. Fix this on CPU: components of x_k are not close to 1.0.
Hint: figure out which elements are farthest from 1.0.
```

### 2. Locate the second CPU bug

Once the first bug is fixed, the second bug becomes visible:

```
$ ONEAPI_DEVICE_SELECTOR=*:cpu ./jacobi-bugged
[SYCL] Using device: [Intel(R) Core(TM) i7-7567U processor] from [Intel(R) OpenCL]
Iteration 0, relative error = 2.71068
Iteration 20, relative error = 1.77663
Iteration 40, relative error = 1.17049
Iteration 60, relative error = 0.869698
Iteration 80, relative error = 0.691864

success; all elements of the resulting vector are close to 1.0.

fail; Bug 2. Fix this on CPU: the relative error (0.579327) is greater than
    the desired tolerance 0.0001 after 100 iterations.
Hint: check the reduction results at several iterations.
Challenge: in the debugger you can simmulate the computation of a reduced
    value by putting a BP inside the corresponding kernel and defining
    a convenience variable. We will compute the reduced value at this
    convenience variable: at each BP hit we update it with a help of "commands"
    command. After the reduction kernel is finished, the convenience
    variable should contain the reduced value.
    See README for details.
```

Once this bug is fixed, while offloading to CPU you receive the correct result:
which are the same as in `jacobi-fixed` for the offload to CPU device. Since we're 
only running on the CPU, we don't see the GPU bug:

```
$ ONEAPI_DEVICE_SELECTOR=*:cpu ./jacobi-fixed
[SYCL] Using device: [Intel(R) Core(TM) i7-7567U processor] from [Intel(R) OpenCL]
Iteration 0, relative error = 2.7581
Iteration 20, relative error = 0.119557
Iteration 40, relative error = 0.0010374

success; all elements of the resulting vector are close to 1.0.
success; the relative error (9.97509e-05) is below the desired tolerance 0.0001 after 51 iterations.
```

#### 3. Run debug on the GPU

Start debugging GPU only after first two bugs are fixed on CPU.
To debug on Intel GPU device, the device must be Level Zero.  The Level
Zero GPU device, if available, is the default choice when no
`ONEAPI_DEVICE_SELECTOR` is specified.  It corresponds to
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.

Bug 3 is immediately hit while offloading to GPU:

```
$ ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./jacobi-bugged
[SYCL] Using device: [Intel(R) Graphics [0x56c1]] from [Intel(R) Level-Zero]

fail; Bug 3. Fix it on GPU. The relative error has invalid value after iteration 0.
Hint 1: inspect reduced error values. With the challenge scenario
    from bug 2 you can verify that reduction algorithms compute
    the correct values inside kernel on GPU. Take into account
    SIMD lanes: on GPU each thread processes several work items
    at once, so you need to modify your commands and update
    the convenience variable for each SIMD lane, e.g. using
    `thread apply :*`.

Hint 2: why don't we get the correct values at the host part of
    the application?
```

Once all three bugs are fixed, the output of the program should be the same
as for `jacobi-fixed` with the offload to GPU device:

```
$ ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./jacobi-fixed
[SYCL] Using device: [Intel(R) Graphics [0x56c1]] from [Intel(R) Level-Zero]
Iteration 0, relative error = 2.7581
Iteration 20, relative error = 0.119557
Iteration 40, relative error = 0.0010374

success; all elements of the resulting vector are close to 1.0.
success; the relative error (9.97509e-05) is below the desired tolerance 0.0001 after 51 iterations.
```

#### Debugging on FPGA Emulation:

While offloading to FPGA emulation device, only first two bugs appear (similar to CPU):

```
$ ONEAPI_DEVICE_SELECTOR=*:fpga ./jacobi-bugged
[SYCL] Using device: [Intel(R) FPGA Emulation Device] from [Intel(R) FPGA Emulation Platform for OpenCL(TM) software]
Iteration 0, relative error = 2.71116
Iteration 20, relative error = 1.70922
Iteration 40, relative error = 1.10961
Iteration 60, relative error = 0.818992
Iteration 80, relative error = 0.648988

fail; Bug 1. Fix this on CPU: components of x_k are not close to 1.0.
Hint: figure out which elements are farthest from 1.0.
```

And after fixing the first bug:

```
$ ONEAPI_DEVICE_SELECTOR=*:fpga ./jacobi-bugged
[SYCL] Using device: [Intel(R) FPGA Emulation Device] from [Intel(R) FPGA Emulation Platform for OpenCL(TM) software]
Iteration 0, relative error = 2.71068
Iteration 20, relative error = 1.77663
Iteration 40, relative error = 1.17049
Iteration 60, relative error = 0.869698
Iteration 80, relative error = 0.691864

success; all elements of the resulting vector are close to 1.0.

fail; Bug 2. Fix this on CPU: the relative error (0.579327) is greater than
    the desired tolerance 0.0001 after 100 iterations.
Hint: check the reduction results at several iterations.
Challenge: in the debugger you can simmulate the computation of a reduced
    value by putting a BP inside the corresponding kernel and defining
    a convenience variable. We will compute the reduced value at this
    convenience variable: at each BP hit we update it with a help of "commands"
    command. After the reduction kernel is finished, the convenience
    variable should contain the reduced value.
    See README for details.
```

After both bugs are fixed, the output of `jacobi-bugged` should become the same as for
`jacobi-fixed`:

```
$ ONEAPI_DEVICE_SELECTOR=*:fpga ./jacobi-fixed
[SYCL] Using device: [Intel(R) FPGA Emulation Device] from [Intel(R) FPGA Emulation Platform for OpenCL(TM) software]
Iteration 0, relative error = 2.7581
Iteration 20, relative error = 0.119557
Iteration 40, relative error = 0.0010374

success; all elements of the resulting vector are close to 1.0.
success; the relative error (9.97509e-05) is below the desired tolerance 0.0001 after 51 iterations.
```

## More Useful Commands

`help <cmd>`
: Print help info about the command `cmd`.

`run [arg1, ... argN]`
: Start the program, optionally with arguments.

`break <filename>:<line>`
: Define a breakpoint at the given source file's specified line.

`info break`
: Show the defined breakpoints.

`delete <N>`
: Remove the `N`th breakpoint.

`watch <exp>`
: Stop when the value of the expression `exp` changes.

`step`, `next`
: Single-step a source line, stepping into/over function calls.

`continue`
: Continue execution.

`print <exp>`
: Print value of expression `exp`.

`backtrace`
: Show the function call stack.

`up`, `down`
: Go one level up/down in the function call stack.

`disassemble`
: Disassemble the current function.

`info args`/`locals`
: Show the arguments/local vars of the current function.

`info reg <regname>`
: Show contents of the specified register.

`info inferiors`
: Display information about the `inferiors`. For GPU offloading, one inferior
  represents the host process, and another (`gdbserver-gt`) represents the
  kernel.

`info threads <ID>`
: Display information about threads with id `ID`, including their active SIMD
  lanes. Omit id to display all threads.

`thread <thread_id>:<lane>`
: Switch context to the SIMD lane `lane` of the specified thread.
  E.g: `thread 2.6:4`

`thread apply <thread_id>:<lane> <cmd>`
: Apply command `cmd` to the specified lane of the thread.
  E.g.: `thread apply 2.3:* print element` prints `element` for each active lane
  of thread 2.3. Useful for inspecting vectorized values.

`x /<format> <addr>`
: Examine the memory at address `addr` according to `format`.
  E.g.: `x /i $pc` shows the instruction pointed by the program counter.
  `x /8wd &count` shows eight words in decimal format located at the address
  of `count`.

`set nonstop on/off`
: Enable/disable the nonstop mode. This command may **not** be used after the
  program has started.

`set scheduler-locking on/step/off`
: Set the scheduler locking mode.

`maint jit dump <addr> <filename>`
: Save the JIT'ed objfile that contains address `addr` into the file `filename`.
  Useful for extracting the kernel when running on the CPU device.

`cond [-force] <N> <exp>`
: Define the expression `exp` as the condition for breakpoint `N`. Use the
  optional `-force` flag to force the condition to be defined even when `exp` is
  invalid for the current locations of the breakpoint. Useful for defining
  conditions involving JIT-produced artificial variables.

---

\* Intel is a trademark of Intel Corporation or its subsidiaries. Other names
and brands may be claimed as the property of others.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third-party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
