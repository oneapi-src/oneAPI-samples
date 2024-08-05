# `Optimization Integral` Sample

The `Optimization Integral` sample is designed to illustrate compiler optimization features and programming concepts.

| Area                     | Description
|:---                      |:---
| What you will learn      | Optimization using the Intel® Fortran Compiler
| Time to complete         | 15 minutes

## Purpose
This sample demonstrates how to use Intel® Fortran Compiler options to optimize applications for performance.

>**Note** Some of these automatic optimizations use features and options that can restrict program execution to specific architectures.

The primary compiler option for optimization is [`-O[n]`](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/o-001.html), where `n` is the optimization level. The following table provides a description 
for each optimization level.

| Linux* Option | Windows* Option  | Description
|:---          |:---             |:---
|`-O0`         |`/Od`            |Disables all optimizations.
|`-O1`         |`/O1`            |Enables optimizations for speed and disables some optimizations that increase code size and affect speed.
|`-O2`         |`/O2`            |Enables optimizations for speed. This is the recommended optimization level and is the default. Vectorization is enabled at `O2` and higher levels.
|`-O3`         |`/O3`            |Performs `O2` optimizations and enables more aggressive loop transformations such as Fusion, Block-Unroll-and-Jam, and collapsing IF statements.

More information about Intel® Fortran Compiler optimization options is available in the [Intel® Fortran Compiler Developer Guide and
Reference](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/optimization-options.html).

## Prerequisites
| Optimized for           | Description
|:---                     |:---
| OS                      | Linux <br> Windows
| Software                | Intel® Fortran Compiler

>**Note** The Intel® Fortran Compiler is part of the [Intel® oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html).

## Key Implementation Details
The sample program computes the integral (area under the curve) of a user-supplied function over an interval in a 
stepwise fashion. The interval is split into segments. At each segment position the area of a rectangle is computed 
with the height of the value of sine at that point. The width is the segment width. The areas of the rectangles are then 
summed. The process repeats with smaller and smaller width rectangles, more closely approximating the true value.

The source code for this program also demonstrates recommended Fortran coding practices.

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the
oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This
practice ensures that your compiler, libraries, and tools are ready for
development.

> **Note** If you have not already done so, set up your CLI environment by
> sourcing  the `setvars` script in the root of your oneAPI installation.
>
> **Linux**
> - For system wide installations in the default installation directory: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
>
> **Windows**
> - Under normal circumstances, you do not need to run the setvars.bat batch file. The terminal shortcuts 
> in the Windows Start menu, Intel oneAPI command prompt for <target architecture> for Visual Studio <year>, 
> set these variables automatically.
>
> - For additional information, see [Use the Command Line on Windows](https://www.intel.com/content/www/us/en/docs/fortran-compiler/get-started-guide/current/get-started-on-windows.html#GUID-A9B4C91D-97AC-450D-9742-9D895BC8AEE1).
>
> For more information on configuring environment variables, see [Use the
> setvars Script with Linux* and Windows*](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/specifying-the-location-of-compiler-components.html).

## Build and Run the `Fortran Optimization Integral` Sample
You will build the program several times with different optimization levels. Notice the timings for each change.

> **Note**
> There are separate sets of instructions for Linux and Windows for each optimization option.

### Build and Run with Optimization Level 0

Optimization level 0 ([`-O0` on Linux or `/Od` on Windows](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/o-001.html)) disables all optimizations.

#### Linux

1. Change to the sample directory.
2. Using your favorite editor, open the `Makefile` file.
3. Uncomment the line for `-O0`.
4. Save the change. Your final version should resemble the following example:
   ```
   FC = ifx -O0
   #FC = ifx -O1
   #FC = ifx -O2
   #FC = ifx -O3 
   ```
5. Compile the program.
   ```
   make
   ```
6. Run the program.
   ```
   make run
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 3.776983 seconds 
   ```
8. Clean the project and program files.
   ```
   make clean
   ```

#### Windows
1. Change to the sample directory.
2. Using your favorite editor, open the `build.bat` file.
3. Uncomment the line for `/Od`.
4. Save the change. Your final version should resemble the following example:
   ```
   ifx /Od src/int_sin.f90 /o int_sin.exe
   : ifx /O1 src/int_sin.f90 /o int_sin.exe
   : ifx /O2 src/int_sin.f90 /o int_sin.exe
   : ifx /O3 src/int_sin.f90 /o int_sin.exe
   ```
5. Compile the program.
   ```
   build.bat
   ```
6. Run the program.
   ```
   run.bat
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 3.776983 seconds 
   ```

### Build and Run with Optimization Level 1

Optimization level 1 (`O1`) enables optimizations for speed and disables some optimizations that increase code size and affect speed. 

#### Linux
1. Change to the sample directory.
2. Using your favorite editor, open the `Makefile` file.
3. Uncomment the line for `-O1`.
4. Save the change. Your final version should resemble the following example:
   ```
   #FC = ifx -O0
   FC = ifx -O1
   #FC = ifx -O2
   #FC = ifx -O3 
   ```
5. Compile the program.
   ```
   make
   ```
6. Run the program.
   ```
   make run
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 1.444569 seconds
   ```
8. Clean the project and program files.
   ```
   make clean
   ```

#### Windows
1. Change to the sample directory.
2. Using your favorite editor, open the `build.bat` file.
3. Uncomment the line for `/O1`.
4. Save the change. Your final version should resemble the following example:
   ```
   : ifx /Od src/int_sin.f90 /o int_sin.exe
   ifx /O1 src/int_sin.f90 /o int_sin.exe
   : ifx /O2 src/int_sin.f90 /o int_sin.exe
   : ifx /O3 src/int_sin.f90 /o int_sin.exe
   ```
5. Compile the program.
   ```
   build.bat
   ``` 
6. Run the program.
   ```
   run.bat
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 1.444569 seconds
   ```

### Build and Run with Optimization Level 2

Optimization level 2 (`O2`) enables optimizations for speed. This is the recommended optimization level and is the default. Vectorization is enabled at level 2 and higher.

#### Linux
1. Change to the sample directory.
2. Using your favorite editor, open the `Makefile` file.
3. Uncomment the line for `-O2`.
4. Save the change. Your final version should resemble the following example:
   ```
   #FC = ifx -O0
   #FC = ifx -O1
   FC = ifx -O2
   #FC = ifx -O3 
   ```
5. Compile the program.
   ```
   make
   ```
6. Run the program.
   ```
   make run
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 0.5143980 seconds
   ```
8. Clean the project and program files.
   ```
   make clean
   ```

#### Windows
1. Change to the sample directory.
2. Using your favorite editor, open the `build.bat` file.
3. Uncomment the line for `/O2`.
4. Save the change. Your final version should resemble the following example:
   ```
   : ifx /Od src/int_sin.f90 /o int_sin.exe
   : ifx /O1 src/int_sin.f90 /o int_sin.exe
   ifx /O2 src/int_sin.f90 /o int_sin.exe
   : ifx /O3 src/int_sin.f90 /o int_sin.exe 
   ```
5. Compile the program.
   ```
   build.bat
   ```
6. Run the program.
   ```
   run.bat
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 0.5143980 seconds 
   ```

### Build and Run with Optimization Level 3

Optimization level 3 (`O3`) performs level 2 optimizations and enables more aggressive loop transformations such as Fusion, Block-Unroll-and-Jam, and collapsing IF statements.

#### Linux
1. Change to the sample directory.
2. Using your favorite editor, open the `Makefile` file.
3. Uncomment the line for `-O3`.
4. Save the change. Your final version should resemble the following example:
   ```
   #FC = ifx -O0
   #FC = ifx -O1
   #FC = ifx -O2
   FC = ifx -O3
   ```
5. Compile the program.
   ```
   make
   ```
6. Run the program.
   ```
   make run
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 0.5133380 seconds
   ```
8. Clean the project and program files.
   ```
   make clean
   ```

#### Windows
1. Change to the sample directory.
2. Using your favorite editor, open the `build.bat` file.
3. Uncomment the line for `/O3`.
4. Save the change. Your final version should resemble the following example:
   ```
   : ifx /Od src/int_sin.f90 /o int_sin.exe
   : ifx /O1 src/int_sin.f90 /o int_sin.exe
   : ifx /O2 src/int_sin.f90 /o int_sin.exe
   ifx /O3 src/int_sin.f90 /o int_sin.exe  
   ```
5. Compile the program.
   ```
   build.bat
   ```
6. Run the program.
   ```
   run.bat
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 0.5133380 seconds 
   ```

### What Changed?
You might have noticed there are big jumps in "CPU Time" when going from `O0` to `O1` and from `O1` to `O2`. There are minimal performance 
gains going from `O2` to `O3`. This varies by application, but generally `O2` has the most useful optimizations when 
using the Intel&reg; Fortran Compiler. Sometimes `O3` can help, but the `O2` option is the appropriate option for most 
applications. 

If no optimization level is specified, the default is `O2`.

### Further Exploration
The Intel® Fortran Compiler has many optimization options. If you have a genuine Intel® processor, there are two additional compiler options you should learn about:

| Linux                    | Windows              | Description                                         |
| :----------              | :----------          | :------                                             |
| `-xhost`                 | `/Qxhost`            | `host` is one of the *code*s of `x`. <br>Tells the compiler to generate instructions for the highest instruction set available on the compilation host processor. The default target architecture supports only Intel® SSE2 instructions. |
| `-align array64byte`     | `/align:array64byte` | Specifies a starting boundary to align arrays.      |
  
Read the *Compiler Options* section of the [Intel® Fortran Compiler Developer Guide and
Reference](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/current/compiler-options-001.html)
 for more information about these and other compiler options.

#### Linux
1. Change to the sample directory.
2. Using your favorite editor, open the `Makefile` file.
3. Uncomment the line for `-O3` and add the two options as shown.
   ```
   #FC = ifx -O0
   #FC = ifx -O1
   #FC = ifx -O2
   FC = ifx -O3 -xhost -align array64byte
   ```
4. Save the change.
5. Compile the program.
   ```
   make
   ```
6. Run the program.
   ```
   make run
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 0.2578490 seconds
   ```
8. Clean the project and program files.
   ```
   make clean
   ```

#### Windows
1. Change to the sample directory. 
2. Using your favorite editor, open the `build.bat` file.
3. Uncomment the line for `/O3` and add the two compiler options as shown.
   ```
   : ifx /Od src/int_sin.f90 /o int_sin.exe
   : ifx /O1 src/int_sin.f90 /o int_sin.exe
   : ifx /O2 src/int_sin.f90 /o int_sin.exe
   ifx /O3 /Qxhost /align:array64byte src/int_sin.f90 /o int_sin.exe
   ```
4. Save the change.
5. Compile the program.
   ```
   build.bat
   ```
6. Run the program.
   ```
   run.bat
   ```
7. Notice the CPU time. Your time should look similar to this example.
   ```
   CPU Time = 0.2578490 seconds
   ```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
