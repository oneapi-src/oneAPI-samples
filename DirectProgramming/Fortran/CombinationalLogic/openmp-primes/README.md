# `OpenMP* Primes` Samples

The `OpenMP* Primes` sample is designed to illustrate how to use the OpenMP* API with the Intel® Fortran Compiler.

This program finds all primes in the first 40,000,000 integers, the number of
4n+1 primes, and the number of 4n-1 primes in the same range. The sample
demonstrates how to use two OpenMP* directives to help speed up code.


| Area                     | Description
|:---                      |:---
| What you will learn      | How to build and run a Fortran OpenMP application using the Intel® Fortran Compiler
| Time to complete         | 10 minutes

## Purpose

This program finds all primes in the first 40,000,000 integers, the number of
4n+1 primes, and the number of 4n-1 primes in the same range. It shows how to use
two OpenMP directives to help speed up the code.

First, a dynamic schedule clause is used with the OpenMP for a directive.
Because the workload of the DO loop increases as its index get bigger, the
default static scheduling does not work well. Instead, dynamic scheduling
accounts for the increased workload. Dynamic scheduling itself has more overhead
than static scheduling, so a chunk size of 10 is used to reduce the overhead for
dynamic scheduling.

Second, a reduction clause is used instead of an OpenMP critical directive to
eliminate lock overhead. Using a critical directive would cause excessive lock
overhead due to the one-thread-at-time update of the shared variables each time
through the DO loop. Instead, the reduction clause causes only one update of the
shared variables once at the end of the loop.

## Prerequisites

| Optimized for            | Description
|:---                      |:---
| OS                       | Linux*<br>Windows*
| Software                 | Intel® Fortran Compiler

>**Note**: The Intel® Fortran Compiler is included in the [Intel® oneAPI HPC
>Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) or available as a
[stand-alone download](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#fortran).

## Key Implementation Details

The Intel Fortran Compiler includes all libraries and headers necessary to
compile and run OpenMP-enabled Fortran applications.

Use the following options to compile the program versions.

- [`-qopenmp` (Linux) or `/Qopenmp` (Windows)](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/qopenmp-qopenmp.html) enables compiler recognition of OpenMP* directives. Omitting this
  option results in a serial program.
- [`-O[n]` (Linux) or `/O[n]` (Windows)](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/o-001.html) sets the optimization level from level 1 (`-O1`) to level 3 (`-O3`). You can disable all optimizations using `-O0` (Linux) or `/Od` (Windows). 

>**Note**: You can find more information about these options in the *Compiler
>Options* section of the [Intel® Fortran Compiler Developer Guide and
>Reference](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/overview.html).

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the
oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This
practice ensures that your compiler, libraries, and tools are ready for
development.

> **Note**: If you have not already done so, set up your CLI environment by
> sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux:
> - For system wide installations in the default installation directory: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
>
> Windows:
> - Under normal circumstances, you do not need to run the setvars.bat batch file. The terminal shortcuts 
> in the Windows Start menu, Intel oneAPI command prompt for <target architecture> for Visual Studio <year>, 
> set these variables automatically.
>
> For additional information, see [Use the Command Line on Windows](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/use-the-command-line-on-windows.html).
>
> For more information on configuring environment variables, see [Use the
> setvars Script with Linux and Windows](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/specifying-the-location-of-compiler-components.html).

## Build the `OpenMP Primes` Sample

1. Change to the sample directory.
2. Build debug and release versions of the program.

   Linux:

   ```
   make clean
   make debug
   make
   ```

   Windows:

   ```
   build.bat
   ```

## Run the `OpenMP* Primes` Program

You can run different versions of the program to discover application runtime
changes.

### Experiment 1: Run the Debug Version

1. Run the program.

   Linux:

   ```
   make debug_run
   ```

   Windows:

   ```
   debug_run.bat
   ```

   Notice the timestamp. With multi-threaded applications, use Elapsed Time to measure the time. CPU time is the time 
   accumulated for all threads.

### Experiment 2: Run the Optimized Version

1. Run the program.
   
   Linux:

   ```
   make run
   ```

   Windows:

   ```
   run.bat
   ```

   Did the debug (unoptimized) version run slower?

### Experiment 3: Change the Number of Threads

By default, an OpenMP application creates and uses as many threads as the number
of "processors" in a system. A "processor" is defined as the number of logical
processors, which are twice the number of physical cores on hyperthreaded cores.

OpenMP uses the environment variable `OMP_NUM_THREADS` to set the number of
threads to use.

1. Experiment with a single thread.

   Linux:

   ```
   export OMP_NUM_THREADS=1`
   make run
   ```

   Windows:

   ```
   set OMP_NUM_THREADS=1
   run.bat
   ```

   Notice the number of threads reported by the application.

2. Experiment with 2 threads.

   Linux:

   ```
   export OMP_NUM_THREADS=2
   make run
   ``` 

   Windows:

   ```
   set OMP_NUM_THREADS=2
   run.bat
   ```

   Notice if the application ran faster with more threads.

3. Experiment with the number of threads and see how changing the number of threads 
   affects performance.

4. On Linux clean the object and executable files.

   ```
   make clean
   ```

## Further Reading

Read about using OpenMP with the Intel® Fortran Compiler in the *OpenMP Support* section of the [Intel® Fortran Compiler
Developer Guide and
Reference](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/overview.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).