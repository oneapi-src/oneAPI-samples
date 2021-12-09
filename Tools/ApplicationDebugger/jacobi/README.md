# `jacobi` Sample

The 'jacobi' code sample is a small DPC++ code sample for exercising application
debugging using Intel&reg; Distribution for GDB\*. It is highly recommended that
you go through this sample *after* you familiarize yourself with the
[array-transform](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/ApplicationDebugger/array-transform)
sample and the
[Get Started Guide](https://software.intel.com/en-us/get-started-with-debugging-dpcpp)
for the debugger.

This sample contains two versions of the same program: `jacobi-bugged` and
`jacobi-fixed`. The latter works correctly, but in the former several bugs are
injected. You can try to find and fix them using the debugger.

| Optimized for       | Description
|---------------------|--------------
| OS                  | Linux Ubuntu 18.04 to 20.04, CentOS* 8, Fedora* 30, SLES 15; Windows* 10
| Hardware            | Kaby Lake with GEN9 (on GPU) or newer (on CPU)
| Software            | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn | Find existing bugs in a program using the debugger
| Time to complete    | 1 hour for CPU or FPGA emulator; 2 hours for GPU

## Purpose

The `jacobi` sample is a DPC++ application that solves a hardcoded linear system
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
devices. For convenience, the `jacobi` code sample provides the ability to
select the target device by using a command-line argument of `cpu`, `gpu`, or
`accelerator`.

For an overview of the Jacobi method please refer to
[Jacobi method|https://en.wikipedia.org/wiki/Jacobi_method].

For an overview of the debugger, please refer to the
[Get Started Guide](https://software.intel.com/en-us/get-started-with-debugging-dpcpp)
of the application debugger.

## Recommended commands

For checking variables values you can use: `print`, `printf`, `display`, `info locals`.

To define specific actions when a BP is hit, use `commands`, e.g.

```
(gdb) commands 1
Type commands for breakpoint(s) 1, one per line.
End with a line saying just "end".
>silent
>print var1
>continue
>end
```

sets actions for breakpoint 1. At each hit of the breakpoint, the variable `var1`
will be printed and then the debugger will continue until the next breakpoint hit.
Here we start the command list with `silent` -- it helps to suppress GDB output
about hitting the BP.

On GPU all threads execute SIMD. To apply the command to every SIMD lane
that hit the BP, add `/a` modifier. In the following we print local variable `gid`
for every SIMD lane that hits the breakpoint at line 130:

```
(gdb) break 130
Breakpoint 1 at 0x4125ac: file jacobi-bugged.cpp, line 130.
(gdb) commands /a
Type commands for breakpoint(s) 1, one per line.
Commands will be applied to all hit SIMD lanes.
End with a line saying just "end".
>print gid
>end
(gdb) running
<... GDB output omitted...>
[Switching to Thread 1.1073741824 lane 0]

Thread 2.1 hit Breakpoint 1, with SIMD lanes [0-7], prepare_for_next_iteration(...arguments omitted...) at jacobi-bugged.cpp:130
130         acc_l1_norm_x_k1 += abs(acc_x_k1[gid]); // Bug 2 challenge: breakpoint here.
$1 = cl::sycl::id<1> = {8}
$2 = cl::sycl::id<1> = {9}
$3 = cl::sycl::id<1> = {10}
$4 = cl::sycl::id<1> = {11}
$5 = cl::sycl::id<1> = {12}
$6 = cl::sycl::id<1> = {13}
$7 = cl::sycl::id<1> = {14}
$8 = cl::sycl::id<1> = {15}
```

You can apply a command to specified threads and SIMD lanes with `thread apply`.
The following command prints `gid` for each active SIMD lane of the current thread:

```
(gdb) thread apply :* -q print gid
$9 = cl::sycl::id<1> = {8}
$10 = cl::sycl::id<1> = {9}
$11 = cl::sycl::id<1> = {10}
$12 = cl::sycl::id<1> = {11}
$13 = cl::sycl::id<1> = {12}
$14 = cl::sycl::id<1> = {13}
$15 = cl::sycl::id<1> = {14}
$16 = cl::sycl::id<1> = {15}
```

`-q` flag suppresses the additional output such as for which thread and lane
the command was executed.

You can set a convenience variable and then modify it:

```
(gdb) set $temp=1
(gdb) print $temp
$1 = 1
(gdb) print ++$temp
$2 = 2
```

If you want to step through the program, use: `step`, `next`. Ensure that
scheduler locking is set to `step` or `on` (otherwise, you will be switched
to a different thread while stepping):

```
(gdb) set scheduler-locking step
```

## Key Implementation Details

Jacobi method is an iterative solver for a system of linear equations. The system
must be strictly diagonally dominant to ensure that the method converges
to the solution. In our case, the matrix is hardcoded into the kernel from
`main_computation` function.

All computations happen inside a while-loop. There are two exit criteria
from the loop. First, if the relative error falls below the desired tolerance
we consider it as a success and the algorithm converged. Second, if we exceed
the maximum number of iterations we consider it as a fail, the algorithm
did not converge.

Each iteration has two parts: `main_computation` and `prepare_for_next_iteration`.

### Main computation

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
The computation of the final relative error happens in `main`:
we divide the absolute error `abs_error` by the L1 norm of x_k1 (`l1_norm_x_k1`).

### Prepare for the next iteration

First, we perform sum-reductions over the vectors `x_k1 - x_k` and `x_k1`
to compute L1 norms of `x_k - x_k1` and `x_k1` respectively.

Second, we update `x_k` with the new value from `x_k1`.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third-party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building and Running the `jacobi` Project

If you have not already done so, set up your CLI environment by sourcing
the setvars script located in the root of your oneAPI installation.

Linux Sudo: `. /opt/intel/oneapi/setvars.sh`

Linux User: `. ~/intel/oneapi/setvars.sh`

Windows: `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`

### Setup

Preliminary setup steps are needed for the debugger to function. Please see the
setup instructions in the Get Started Guide based on your OS:
[Linux](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-linux),
[Windows](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-windows).


### Include Files

The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on
your development system.


### Running Samples In DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the
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

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the Extension Pack for Intel® oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System

Perform the following steps:

1.  Build the project using the following `cmake` commands.

    ```
    $ cd jacobi
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```

    This builds both `jacobi-bugged` and `jacoby-fixed` versions of the program.
    > Note: The cmake configuration enforces the `Debug` build type.

2.  Run the buggy program:

    ```
    $ ./jacobi-bugged <device>
    ```

    > Note: `<device>` is the type of the device to offload the kernel.
    > Use `cpu`, `gpu`, or `accelerator` to select the CPU, GPU, or the
    > FPGA emulator device, respectively.  E.g.:

    ```
    $ ./jacobi-bugged cpu
    ```

3.  Start a debugging session:

    ```
    $ gdb-oneapi --args jacobi-bugged <device>
    ```

4.  Clean the program using:

    ```
    $ make clean
    ```


For instructions about starting and using the debugger, please see the
[Get Started Guide (Linux)](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-linux).

### On a Windows* System Using Visual Studio* Version 2017 or Newer

#### Command line using MSBuild

* `set CL_CONFIG_USE_NATIVE_DEBUGGER=1`
* `MSBuild lacobi.sln /t:Rebuild /p:Configuration="debug"`

#### Visual Studio IDE

1. Right-click on the solution files and open via either Visual Studio 2017 or
   in 2019.

2. Select Menu "Build > Build Solution" to build the selected configuration.

3. Select Menu "Debug > Start Debugging" to run the program, the default startup
   project is `jacobi-bugged`.

4. The solution file is configured to pass `cpu` as the argument to the program
   while using "Local Windows Debugger", and `gpu` while using "Remote Windows
   Debugger". To select a different device, go to the project's "Configuration
   Properties > Debugging" and set the "Command Arguments" field. Use `gpu` or
   `accelerator` to target the GPU or the FPGA emulator device, respectively.

For detailed instructions about starting and using the debugger, please see the
[Get Started Guide (Windows)](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-windows).

### Example Outputs

The selected device is displayed in the output.

#### No device specified

If no device is specified both programs `jacobi-bugged` and `jacobi-fixed` return an error:

```
$ ./jacobi-bugged
Usage: ./jacobi-bugged <host|cpu|gpu|accelerator>
```

```
$ ./jacobi-fixed
Usage: ./jacobi-fixed <host|cpu|gpu|accelerator>
```

#### CPU

When run the original `jacobi-bugged`, the program shows the first bug:

```
$ ./jacobi-bugged cpu
[SYCL] Using device: [Intel(R) Core(TM) i7-7567U processor] from [Intel(R) OpenCL]
Iteration 0, relative error = 2.71116
Iteration 20, relative error = 1.70922
Iteration 40, relative error = 1.10961
Iteration 60, relative error = 0.818992
Iteration 80, relative error = 0.648988

fail; Bug 1. Fix this on CPU: components of x_k are not close to 1.0.
Hint: figure out which elements are farthest from 1.0.
```

Once the first bug is fixed, the second bug becomes visible:

```
$ ./jacobi-bugged cpu
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
which are the same as in `jacobi-fixed` for the offload to CPU device:

```
$ ./jacobi-fixed cpu
[SYCL] Using device: [Intel(R) Core(TM) i7-7567U processor] from [Intel(R) OpenCL]
Iteration 0, relative error = 2.7581
Iteration 20, relative error = 0.119557
Iteration 40, relative error = 0.0010374

success; all elements of the resulting vector are close to 1.0.
success; the relative error (9.97509e-05) is below the desired tolerance 0.0001 after 51 iterations.
```

#### GPU

We advise to start debugging GPU only after first two bugs are fixed on CPU.

Bug 3 is immediately hit while offloading to GPU:

```
$ ./jacobi-bugged gpu
[SYCL] Using device: [Intel(R) Iris(R) Plus Graphics 650 [0x5927]] from [Intel(R) Level-Zero]

fail; Bug 3. Fix it on GPU. The relative error has invalid value after iteration 0.
Hint 1: inspect reduced error values. With the challenge scenario
    from bug 2 you can verify that reduction algorithms compute
    the correct values inside kernel on GPU. Take into account
    SIMD lanes: on GPU each thread processes several work items
    at once, so you need to modify your commands and update
    the convenience variable for each SIMD lane.
Hint 2: why don't we get the correct values at the host part of
    the application?
```

Once all three bugs are fixed, the output of the program should be the same
as for `jacobi-fixed` with the offload to GPU device:

```
$ ./jacobi-fixed gpu
[SYCL] Using device: [Intel(R) Iris(R) Plus Graphics 650 [0x5927]] from [Intel(R) Level-Zero]
Iteration 0, relative error = 2.7581
Iteration 20, relative error = 0.119557
Iteration 40, relative error = 0.0010374

success; all elements of the resulting vector are close to 1.0.
success; the relative error (9.97509e-05) is below the desired tolerance 0.0001 after 51 iterations.
```

#### FPGA Emulation:

While offloading to FPGA emulation device, only first two bugs appear (similar to CPU):

```
$ ./jacobi-bugged accelerator
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
$ ./jacobi-bugged accelerator
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
$ ./jacobi-fixed accelerator
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
  Useful for extracting the DPC++ kernel when running on the CPU device.

`cond [-force] <N> <exp>`
: Define the expression `exp` as the condition for breakpoint `N`. Use the
  optional `-force` flag to force the condition to be defined even when `exp` is
  invalid for the current locations of the breakpoint. Useful for defining
  conditions involving JIT-produced artificial variables.
  E.g.: `cond -force 1 __ocl_dbg_gid0 == 19`.

---

\* Intel is a trademark of Intel Corporation or its subsidiaries. Other names
and brands may be claimed as the property of others.
