# `Tachyon` Sample

The `Tachyon` sample shows how to improve the performance of serial programs by using parallel processing with OpenMP* or Intel® Threading Building Blocks (Intel® TBB). The `Tachyon` sample is an implementation of the tachyon program; it is a 2-D raytracer program that renders objects described in data files. 

| Area                              | Description
|:---                               |:---
| What you will learn               | How to implement parallelization using OpenMP or Intel® Threading Building Blocks (Intel® TBB)  
| Time to complete                  | 15 minutes

## Purpose

The `Tachyon` sample shows how to use OpenMP or Intel® oneAPI Threading Building Blocks (oneTBB) to improve the performance of serial applications by using parallel processing. 

The sample starts with a serial CPU implementation of the tachyon program and shows how to use OpenMP or Intel® oneTBB to implement effective threading in the program. 

The sample application displays the execution time required to render the object. This time is an indication of the speedup obtained with parallel implementations compared to a baseline established with the initial serial implementation.

The sample produces the following image:  </br> ![image](https://user-images.githubusercontent.com/111458217/186752964-af23ce82-9e4d-4bec-b60f-a5fd427741f8.png)

Five versions of the tachyon program are included in the sample.

- `build_serial.cpp`: basic serial CPU implementation.
- `build_with_openmp.cpp`: basic OpenMP version that uses OpenMP to divide work across threads, but is not optimized.
- `build_with_openmp_optimized.cpp`: optimized OpenMP version that improves the threading implementation.
- `build_with_tbb.cpp`: basic Intel® oneTBB version that uses Intel® oneTBB to divide the work across threads, but is not optimized.
- `build_with_tbb_optimized.cpp`: optimized Intel® oneTBB version that improves the threading implementation.

The time to generate the image varies according to the parallel scheduling method used in each version of the program.

<span style="color:red">**K+J TODO: --> Finalize the wording around rendering in Lin/Win**</span>

## Prerequisites

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04 <br>Windows* 11
| Hardware                          | Intel&reg; CPU
| Software                          | Intel® oneAPI DPC++/C++ Compiler<br>Intel oneAPI Threading Building Blocks (oneTBB) <br><br>For Linux the `libXext.so` and `libX11.so` libraries must be installed to display the rendered graphic.



## Key Implementation Details

The sample implements the following OpenMP and oneTBB features.

OpenMP:

<span style="color:red">**TODO --> add key features of implementation**</span>

TBB: 

<span style="color:red">**TODO --> add key features of implementation**</span>

>**Note**: For comprehensive information about oneAPI programming, see the *[Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide)*. (Use search or the table of contents to find relevant information quickly.)


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.


## Build the Tachyon Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

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

<span style="color:red">**CHECK --> this still valid/tested?**</span>

1. Change to the sample directory.
2. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
   

To build a specific version of the program, run cmake with the version name. For example: 

```
make tachyon.serial
make tachyon.openmp
make tachyon.openmp_optimized
make tachyon.tbb
make tachyon.tbb_optimized
```

### On Windows

**Using Visual Studio***

Build the program using Visual Studio 2019 or newer.

1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. Right-click on the project in **Solution Explorer** and select **Rebuild**.
4. From the top menu, select **Debug** > **Start without Debugging**. (This runs the program.)

**Using MSBuild**

1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild "tachyon_samples_2022.sln" /t:Rebuild /p:Configuration="Release"`

  > **Note**: Remember to use Release mode for better performance.

> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Guided Walkthrough


This guided walkthrough starts with the serial implementation to establish a performance baseline and then compares the baseline to the OpenMP and TBB versions of the program with optimizations to the code.

### Run the Serial Version

Run the serial version of the program to establish a baseline execution time.  

#### On Linux

Run the serial version of the program to establish a baseline execution time.  

1. Change to the build directory in your sample directory.
2. Run the executable, providing the balls.dat data file. 

   ```
   ./tachyon.serial ../dat/balls.dat 
   ```

   You will see the following output: 

   ```
   Scene contains 7386 bounded objects. 
   tachyon.serial ../dat/balls.dat: 26.128 seconds 
   ```

#### On Windows

<span style="color:red">**TODO -->**</span>

### Run the OpenMP Versions

Compare the code in `tachyon.openmp.cpp` to `tachyon.serial.cpp`. `tachyon.openmp.cpp` uses the OpenMP library to divide the work among threads.

#### On Linux

1. Change to the build directory in your sample directory. 
2. Run the basic OpenMP version.  

   ```
   ./tachyon.openmp ../dat/balls.dat 
   ```
   
   You will see the following output: 
 
   ```
   Scene contains 7386 bounded objects. 
   tachyon.openmp ../dat/balls.dat: 19.647 seconds 
   ```
3. Run optimized OpenMP version. 

   ```
   ./tachyon.openmp_solution ../dat/balls.dat 
   ```
   
   You will see the following output: 

   ```
   Scene contains 7386 bounded objects. 
   tachyon.openmp_solution ../dat/balls.dat: 2.992 seconds    ```

4. Compare the render time between the basic OpenMP and optimized OpenMP versions. The optimized version shows an improvement in render time. 

<span style="color:red">

*The `build_with_openmp_optimized` project adds dynamic scheduling to the omp pragma, which allows early-finishing threads to take on additional work from slower threads.*

</span>

#### On Windows

<span style="color:red">**TODO -->**</span>

### Run the Intel® oneAPI Threading Building Blocks Versions

Run the Intel® oneAPI Threading Building Blocks (Intel® oneTBB) versions of the program. Before running the Intel® oneTBB version, compare tachyon.tbb.cpp to tachyon.serial.cpp. Note that the Intel® oneTBB version uses the Intel® oneTBB library to divide the work across threads.

#### On Linux

1. Change to the build directory in your sample directory. 
2. Run the basic Intel TBB version. 
 
   ```
   ./tachyon.tbb ../dat/balls.dat 
   ```
   
   You will see the following output: 
 
   ``` 
   Scene contains 7386 bounded objects. 
   tachyon.tbb ../dat/balls.dat: 29.682 seconds 
   ```
 
3. Run the optimized Intel® TBB version. 

   ```
   ./tachyon.tbb_solution ../dat/balls.dat 
   ```
   
   You will see the following output: 
 
   ``` 
   Scene contains 7386 bounded objects. 
   tachyon.tbb_solution ../dat/balls.dat: 2.953 seconds 
   ```

4. Compare the render time between the basic Intel TBB and optimized Intel TBB versions. The optimized version shows an improvement in render time. 

The basic version of the Intel TBB program uses the TBB version of a mutex lock, which prevents multiple threads from working on the code at the same time. In comparison, the optimized version of the TBB program removes mutex lock to allow all threads to work concurrently. 

#### On Windows

<span style="color:red">**TODO: -->**</span>


## License
<p>
    <I>        
        This example includes software developed by John E. Stone.  See
        <A HREF="http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/">here</A> for license information.
    </I>
</p>

