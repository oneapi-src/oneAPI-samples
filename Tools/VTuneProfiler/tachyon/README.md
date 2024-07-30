# `Tachyon` Sample

The `Tachyon` sample shows how to improve the performance of serial programs by using parallel processing with OpenMP* or Intel® oneAPI Threading Building Blocks (Intel® oneTBB). The `Tachyon` sample is an implementation of the tachyon program; it is a 2-D raytracer program that renders objects described in data files. 

| Area                              | Description
|:---                               |:---
| What you will learn               | How to implement parallelization using OpenMP or Intel® oneAPI Threading Building Blocks (Intel® oneTBB)  
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

## Prerequisites

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04 <br>Windows* 11
| Hardware                          | Intel&reg; CPU
| Software                          | Intel® oneAPI DPC++/C++ Compiler<br>Intel oneAPI Threading Building Blocks (oneTBB) <br>Intel VTune&trade; Profiler<br><br>For Linux the `libXext.so` and `libX11.so` libraries must be installed to display the rendered graphic.



## Key Implementation Details

The sample implements the following OpenMP and oneTBB features.

OpenMP:

Uses the **omp parallel for** pragma to thread the horizontal rendering of pixels. 

oneTBB: 

Uses the **tbb::parallel_for** function to thread the horizontal rendering of pixels.

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

Build the program using Visual Studio 2017 or newer.

1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. Right-click on the project in **Solution Explorer** and select **Rebuild**.
4. From the top menu, select **Debug** > **Start without Debugging**. (This runs the program.)

**Using MSBuild**

1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild "tachyon_sample.sln" /t:Rebuild /p:Configuration="Release"`

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


This guided walkthrough starts with the serial implementation to establish a performance baseline and then compares the baseline to the OpenMP and oneTBB versions of the program with optimizations to the code.

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
   tachyon.serial ../dat/balls.dat: 3.706 seconds 
   ```

#### On Windows

Run the build_serial project in VS, or run build_serial.exe ..\dat\balls.dat directly. The Windows version may not show the elapsed execution time, but you can get a high-level look at this implementation's performance by running a hotspots analysis with Intel VTune&trade; Profiler. For more information on Intel VTune Profiler: https://www.intel.com/content/www/us/en/docs/vtune-profiler/get-started-guide/2023/windows-os.html

The Summary view of a hotspots collection with user-mode sampling shows an elapsed time of 3.937s:

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/2394ec2f-c189-4323-b1f0-a77642ec8112)

The CPU Utilization histogram shows that a maximum of one CPU was used throughout the duration of the collection, which is expected in the serial implementation.

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
   tachyon.openmp ../dat/balls.dat: 0.283 seconds 
   ```
3. Run optimized OpenMP version. 

   ```
   ./tachyon.openmp_optimized ../dat/balls.dat
   ```
   
   You will see the following output: 

   ```
   Scene contains 7386 bounded objects. 
   tachyon.openmp_optimized ../dat/balls.dat: 0.153 seconds
   ```

4. Compare the render time between the basic OpenMP and optimized OpenMP versions. The optimized version shows an improvement in render time. 

<span style="color:red">

*The `build_with_openmp_optimized` project adds dynamic scheduling to the omp pragma, which allows early-finishing threads to take on additional work.*

</span>

#### On Windows

1. First run the basic OpenMP implementation. Run the build_with_openmp project in VS, or run **build_with_openmp.exe ..\..\dat\balls.dat** directly. There should be a noticeable improvement in the rendering time. To see how threads performed with the omp parallel for pragma, run a hotspots analysis with Intel VTune Profiler:
   
![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/1e3e15b1-045e-4714-b8c2-a716454066b2)

The elapsed time is almost half of the serial implementation at 2.021s. The CPU histogram still shows a large amount of time where only one CPU was utilized, but the application does make simultaneous use of up to 14 CPUs. The Bottom-up tab has a timeline view which shows thread behavior over time:

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/204b7558-6a3c-4628-bcd5-213f41fd9007)

The first 1.3 seconds are spent reading the data and preparing for rendering. The primary OpenMP thread begins spawning workers near the 1.5 mark, and the timeline shows that some theads finish early and wait for slower threads to finish. Allowing the faster threads to take on more work instead of waiting should result in better performance.

2. Test the optimized OpenMP implementation, which adds dynamic scheduling to the **omp parallel for** pragma. Run the build_with_openmp_optimized project in VS, or run **build_with_openmp_optimized.exe ..\..\dat\balls.dat** directly. The difference in elapsed time may not be noticeable for this quick application, but running a hotspots analysis with Intel VTune Profiler will show if performance improved and by how much.

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/5fc11663-8f7e-4138-8a4e-79260f462b90)

The elapsed time for the optimized version is 1.875s, which is a small improvement. Parallelism has improved marginally, with the application spending a small amount of time utilizing all 16 available CPUs. The timeline view in the Bottom-up tab will show how adding dynamic scheduling changed the behavior of the OpenMP threads:

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/1e63669c-74e4-4466-bce0-5095d9d9fbd1)

The threads now complete at the same time, although there is some spinning as one thread completes its work. 

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
   tachyon.tbb ../dat/balls.dat: 6.504 seconds 
   ```
 
3. Run the optimized Intel® oneTBB version. 

   ```
   ./tachyon.tbb_optimized ../dat/balls.dat 
   ```
   
   You will see the following output: 
 
   ``` 
   Scene contains 7386 bounded objects. 
   tachyon.tbb_optimized ../dat/balls.dat: 0.158 seconds 
   ```

4. Compare the render time between the basic Intel oneTBB and optimized Intel oneTBB versions. The optimized version shows an improvement in render time. 

The basic version of the Intel oneTBB program uses the oneTBB version of a mutex lock, which prevents multiple threads from working on the code at the same time. In comparison, the optimized version of the oneTBB program removes mutex lock to allow all threads to work concurrently. 

#### On Windows
1. Run the build_with_tbb project in VS or run **build_with_tbb.exe ..\..\dat\balls.dat** directly. Running a hotspots analysis with the Intel VTune Profiler shows worse performance than the serial implementation, with an elapsed time of 5.290s:

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/b2029a5b-5862-4045-93f8-ac5bde4c10a6)

The summary view shows a large amount of spin time, with the most active function being a oneTBB lock. The bottom-up view shows several worker threads, but they spend the majority of time spinning:

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/7ad8c037-8999-4e78-96f4-c67f86a4091e)

Taking a closer look at the mutex lock reveals that it is not necessary, as the data is not shared between threads. The optimized implementation removes this mutex.

2. Run the build_with_tbb_optimized project in VS or run **build_with_tbb_optimized.exe ..\..\dat\balls.dat** directly. A hotspots analysis of this implementation shows much better performance:

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/23611d00-3c9a-4915-8d42-925b631d8a7a)

The bottom-up view shows a thread timeline similar to the dynamic openmp implementation, as oneTBB uses work-stealing to balance thread loads:

![image](https://github.com/jenniferdimatteo/oneAPI-samples/assets/32850114/c1b2eb3d-923b-4a45-9872-f2fed890634a)


## License
<p>
    <I>        
        This example includes software developed by John E. Stone.  See
        <A HREF="http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/">here</A> for license information.
    </I>
</p>

