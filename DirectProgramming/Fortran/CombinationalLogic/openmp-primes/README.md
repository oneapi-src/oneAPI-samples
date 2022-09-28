# `OpenMP* Primes` Samples
The `OpenMP* Primes` sample is designed to illustrate how to use the OpenMP* API
with the Intel® Fortran Compiler.

This program finds all primes in the first 40,000,000 integers, the number of
4n+1 primes, and the number of 4n-1 primes in the same range. The sample
illustrates two OpenMP* directives to help speed up code.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to build and run a Fortran OpenMP application using Intel® Fortran Compiler
| Time to complete         | 10 minutes

## Purpose
This program finds all primes in the first 40,000,000 integers, the number of
4n+1 primes, and the number of 4n-1 primes in the same range. It illustrates two
OpenMP* directives to help speed up the code.

First, a dynamic schedule clause is used with the OpenMP* for a directive.
Because the workload of the DO loop increases as its index get bigger, the
default static scheduling does not work well. Instead, dynamic scheduling
accounts for the increased workload. Dynamic scheduling itself has more overhead
than static scheduling, so a chunk size of 10 is used to reduce the overhead for
dynamic scheduling.

Second, a reduction clause is used instead of an OpenMP* critical directive to
eliminate lock overhead. Using a critical directive would cause excessive lock
overhead due to the one-thread-at-time update of the shared variables each time
through the DO loop. Instead, the reduction clause causes only one update of the
shared variables once at the end of the loop.

## Prerequisites
| Optimized for            | Description
|:---                      |:---
| OS                       | macOS* <br> Xcode*
| Software                 | Intel® Fortran Compiler

>**Note**: The Intel® Fortran Compiler is included in the [Intel® oneAPI HPC
>Toolkit (HPC
>Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html).

## Key Implementation Details
The Intel® Fortran Compiler includes all libraries and headers necessary to
compile and run OpenMP* enabled Fortran applications.

You must use the following options to compile the program versions.
- `-qopenmp` enables compiler recognition of OpenMP* directives. (Omitting this
  option results in a serial program.)
- `-fpp` enables the Fortran preprocessor.

You can compile the program with all optimizations disabled using the `-O0` or
at any level of optimization `-O1`, `-O2`, or `-O3`.

>**Note**: You can find more information about these options in the *Compiler
>Options* section of the [Intel® Fortran Compiler Developer Guide and
>Reference](https://www.intel.com/content/www/us/en/develop/documentation/fortran-compiler-oneapi-dev-guide-and-reference).

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the
oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This
practice ensures that your compiler, libraries, and tools are ready for
development.

## Build the `OpenMP* Primes` Sample
> **Note**: If you have not already done so, set up your CLI environment by
> sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux and macOS*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use commands similar to the following: `bash
>   -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the
> setvars Script with Linux* or
> macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### Use Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment
    Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel®
    oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment,
see the [Using Visual Studio Code with Intel® oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On macOS*
1. Change to the sample directory.
2. Build release and debug versions of the program.
   ```
   make clean
   make debug
   ```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the
**Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility
provides configuration and system checks to help find missing dependencies,
permissions errors, and other issues. See the [Diagnostics Utility for Intel®
oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
for more information on using the utility.

## Run the `OpenMP* Primes` Programs
You can run different versions of the program to discover application runtime
changes.

### Experiment 1: Run the Debug Version
1. Run the program.
   ```
   make debug_run
   ```
   Notice the speed.

### Experiment 2: Run the Optimized Version
1. Build and run the release version.
   ```
   make
   ```
2. Run the program.
   ```
   make run
   ```
   Did the debug (unoptimized) version run slower?

### Experiment 3: Change the Number of Threads
By default, an OpenMP application creates and uses as many threads as the number
of  "processors" in a system.  A "processor" is defined as the number of logical
processors, which are twice the number of physical cores on hyperthreaded cores.

OpenMP uses the environment variable `OMP_NUM_THREADS` to set the number of
threads to use.

1. Experiment with a single thread.
   ```
   export OMP_NUM_THREADS=1
   make run
   ```
   Notice the number of threads reported by the application.

2. Experiment with 2 threads.
   ```
   export OMP_NUM_THREADS=2
   make run
   ```
   Notice if the application ran faster with more threads.

3. Experiment with the number of threads, and see changing threads numbers
   affects performance.

4. Clean the project files.
   ```
   make clean
   ```

## Further Reading
Interested in learning more?  Read about using OpenMP with the Intel® Fortran
Compiler in the *OpenMP Support* section of the [Intel® Fortran Compiler
Developer Guide and
Reference](https://www.intel.com/content/www/us/en/develop/documentation/fortran-compiler-oneapi-dev-guide-and-reference).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).