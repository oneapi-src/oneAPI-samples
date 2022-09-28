# `Optimization Integral` Sample
The `Optimization Integral` sample is designed to illustrate compiler optimization features and programming concepts.

| Area                     | Description
|:---                      |:---
| What you will learn      | Optimization using the Intel® Fortran compiler
| Time to complete         | 15 minutes

## Purpose
This sample demonstrates how to use the Intel® Fortran Compiler to optimize applications for performance.

>**Note**: Some of these automatic optimizations use features and options that can restrict program execution to specific architectures.

The primary compiler option is `-O` followed by a numeric optimization "level" from **0**, requesting no optimization, to **3**, which requests all compiler optimizations for the application. The table includes a summary for each optimization option.

| Option       | Description
|:---          |:---
|`-O0`         |Disables all optimizations.
|`-O1`         |Enables optimizations for speed and disables some optimizations that increase code size and affect speed.
|`-O2`         |Enables optimizations for speed. This is the recommended optimization level. Vectorization is enabled at `-O2` and higher levels.
|`-O3`         |Performs `-O2` optimizations and enables more aggressive loop transformations such as Fusion, Block-Unroll-and-Jam, and collapsing IF statements.

Read the *Compiler Options* section of the [Intel® Fortran Compiler Developer Guide and Reference](https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top.html) for more information about these options.

## Prerequisites
| Optimized for           | Description
|:---                     |:---
| OS                      | macOS* <br> Xcode*
| Software                | Intel® Fortran Compiler

>**Note**: The Intel® Fortran Compiler is part of the Intel® oneAPI HPC Toolkit (HPC Kit).

## Key Implementation Details
The sample program computes the integral (area under the curve) of a user-supplied function over an interval in a stepwise fashion. The interval is split into segments. At each segment position the area of a rectangle is computed with the height of sine's value at that point. The width is the segment width. The areas of the rectangles are then summed. The process repeats with smaller and smaller width rectangles, more closely approximating the true value.

The source code for this program also demonstrates recommended Fortran coding practices.

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Fortran Optimization` Sample
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On macOS*
You will build the program several times with different optimization levels. Notice the results for each change.

#### Build with `-O0` Option
1. Change to the sample directory.
2. Using your favorite editor, open the `Makefile` file.
3. Uncomment the line for `-O0`.
4. Save the change. Your final version should resemble the following example.
   ```
   FC = ifort -O0
   #FC = ifort -O1
   #FC = ifort -O2
   #FC = ifort -O3
   ```
5. Build and run the program.
   ```
   make
   make run
   ```
6. Notice the runtime. Your runtime should look similar to this example.
   ```fortran
   CPU Time = 3.776983 seconds
   ```
7. Clean the project and program files.
   ```
   make clean
   ```

#### Build with `-O1` Option
1. Using your favorite editor, open the `Makefile` file.
2. Uncomment the line for `-O1`. 
3. Save the change. Your final version should resemble the following example.
   ```
   #FC = ifort -O0
   FC = ifort -O1
   #FC = ifort -O2
   #FC = ifort -O3
   ```
4. Build and run the program.
   ```
   make
   make run
   ```
5. Notice the runtime. Your runtime should look similar to this example.
   ```fortran
   CPU Time = 1.444569 seconds
   ```
6. Clean the project and program files.
   ```
   make clean
   ```

#### Build with `-O2` Option
1. Using your favorite editor, open the `Makefile` file.
2. Uncomment the line for `-O2`. 
3. Save the change. Your final version should resemble the following example.
   ```
   #FC = ifort -O0
   #FC = ifort -O1
   FC = ifort -O2
   #FC = ifort -O3
   ```
4. Build and run the program.
   ```
   make
   make run
   ```
5. Notice the runtime. Your runtime should look similar to this example.
   ```fortran
   CPU Time = 0.5143980 seconds
   ```
6. Clean the project and program files.
   ```
   make clean
   ```
#### Build with `-O3` Option
1. Using your favorite editor, open the `Makefile` file.
2. Uncomment the line for `-O3`. 
3. Save the change. Your final version should resemble the following example.
   ```
   #FC = ifort -O0
   #FC = ifort -O1
   #FC = ifort -O2
   FC = ifort -O3
   ```
4. Build and run the program.
   ```
   make
   make run
   ```
5. Notice the runtime. Your runtime should look similar to this example.
   ```fortran
   CPU Time = 0.5133380 seconds
   ```
6. Clean the project and program files.
   ```
   make clean
   ```

#### What Changed?
You might have noticed there are big jumps when going from `-O0` to `-O1` and from `-O1` to `-O2`. There are minimal performance gains going from `-O2` to `-O3`. This varies by application, but generally `-O2` has the most useful optimizations when using the Intel&reg; Fortran Compiler. Sometimes `-O3` can help, but the `-O2` option is the appropriate option for most applications.

### Further Exploration
The Intel® Fortran Compiler has many optimization options. If you have a genuine Intel® processor, there are two additional compiler options you should learn about:

- `-xhost` (sub option of `-X` option)
- `-align` **array64byte**
  
Read the *Compiler Options* section of the [Intel® Fortran Compiler Developer Guide and Reference](https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top.html) for more information about these options.

1. Using your favorite editor, open the `Makefile` file.
2. Uncomment `FC = ifort -O3`, and add the two options shown below.
   ```
   #FC = ifort -O0
   #FC = ifort -O1
   #FC = ifort -O2
   FC = ifort -O3 -xhost -align array64byte
   ```
3. Save the change.
4. Build and run the program.
   ```
   make
   make run
   ```
5. Notice the runtime. Your runtime should look similar to this example.
   ```fortran
   CPU Time = 0.2578490 seconds
   ```
6. Clean the project and program files.
   ```
   make clean
   ```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).