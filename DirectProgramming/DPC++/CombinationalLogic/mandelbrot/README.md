# `Mandelbrot` Sample

Mandelbrot is an infinitely complex fractal patterning that is derived from a simple formula.  It demonstrates using DPC++ for offloading computations to a GPU (or other devices) and shows how processing time can be optimized and improved with parallelism.

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to offload the computation to GPU using the Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

## Purpose
Mandelbrot is a DPC++ application that generates a fractal image by initializing a matrix of 512 x 512, where the computation at each point (pixel) is entirely independent of the computation at other points. The sample includes both parallel and serial calculation of the set, allowing for a direct comparison of results. The parallel implementation can demonstrate the use of Unified Shared Memory (USM) or buffers. You can modify parameters such as the number of rows, columns, and iterations to evaluate the difference in performance and load between USM and buffers. This is further described at the end of this document in the "Running the Sample" section.

The code will attempt to execute on an available GPU and fallback to the system's CPU if a compatible GPU is not detected.  The compilation device is displayed in the output along with elapsed time to render the Mandelbrot image. This helps compare different offload implementations based on the complexity of the computation.

## Key Implementation Details
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `Mandelbrot` Program for CPU and GPU

> Note: if you have not already done so, set up your CLI
> environment by sourcing  the setvars script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
> Linux User: . ~/intel/oneapi/setvars.sh
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### Running Samples In DevCloud
Running samples in the Intel DevCloud requires you to specify a compute node. For specific instructions, jump to [Run the sample in the DevCloud](#run-on-devcloud)


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands.
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

> Note: by default, executables are created for both USM and buffers. You can build individually with the following:
>    Create buffers executable: make mandelbrot
>    Create USM executable: make mandelbrot_usm

2. Run the program (default uses buffers):
    ```
    make run
    ```
> Note: for USM use `make run_usm`

3. Clean the program using:
    ```
    make clean
    ```

### On a Windows* System Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild mandelbrot.sln /t:Rebuild /p:Configuration="Release"`


## Running the Sample
### Application Parameters
You can modify the Mandelbrot parameters from within mandel.hpp. The configurable parameters include:
    row_size =
    col_size =
    max_iterations =
    repetitions =
The default row and column size are 512.  Max interatins and repetitions are both 100.  By adjusting the parameters, you can observe how the performance varies using the different offload techniques.
> Note: If the values drop below 128 for row and column, the output is limited to just text in the output window.

### Example of Output
```
Platform Name: Intel(R) OpenCL HD Graphics
  Platform Version: OpenCL 2.1
       Device Name: Intel(R) Gen9 HD Graphics NEO
    Max Work Group: 256
 Max Compute Units: 24

Parallel Mandelbrot set using buffers.
Rendered image output to file: mandelbrot.png (output too large to display in text)
       Serial time: 0.0430331s
     Parallel time: 0.00224131s
Successfully computed Mandelbrot set.
```
### Running the sample in the DevCloud<a name="run-on-devcloud"></a>
1.  Open a terminal on your Linux system.
2.	Log in to DevCloud.
```
ssh devcloud
```
3.	Download the samples.
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

4. Change directories to the  ISO3DFD OpenMP Offload sample directory.
```
cd ~/oneAPI-samples/DirectProgramming/DPC++/CombinationalLogic/mandelbrot
```
#### Build and run the sample in batch mode
The following describes the process of submitting build and run jobs to PBS.
A job is a script that is submitted to PBS through the qsub utility. By default, the qsub utility does not inherit the current environment variables or your current working directory. For this reason, it is necessary to submit jobs as scripts that handle the setup of the environment variables. In order to address the working directory issue, you can either use absolute paths or pass the -d \<dir\> option to qsub to set the working directory.

#### Create the Job Scripts
1.	Create a build.sh script with your preferred text editor:
```
nano build.sh
```
2.	 Add this text into the build.sh file:
```
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
mkdir build
cd build
cmake ..
make
```

3.	Save and close the build.sh file.

4.	Create a run.sh script with with your preferred text editor:
```
nano run.sh
```

5.	 Add this text into the run.sh file:
```
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
cd build
make run
```
6.	Save and close the run.sh file.

#### Build and run
Jobs submitted in batch mode are placed in a queue waiting for the necessary resources (compute nodes) to become available. The jobs will be executed on a first come basis on the first available node(s) having the requested property or label.
1.	Build the sample on a gpu node.

```
qsub -l nodes=1:gpu:ppn=2 -d . build.sh
```

Note: -l nodes=1:gpu:ppn=2 (lower case L) is used to assign one full GPU node to the job.
Note: The -d . is used to configure the current folder as the working directory for the task.

2.	In order to inspect the job progress, use the qstat utility.
```
watch -n 1 qstat -n -1
```
Note: The watch -n 1 command is used to run qstat -n -1 and display its results every second.

3.	When the build job completes, there will be a build.sh.oXXXXXX file in the directory. After the build job completes, run the sample on a gpu node:
```
qsub -l nodes=1:gpu:ppn=2 -d . run.sh
```
4.	When a job terminates, a couple of files are written to the disk:

    <script_name>.sh.eXXXX, which is the job stderr

    <script_name>.sh.oXXXX, which is the job stdout

    Here XXXX is the job ID, which gets printed to the screen after each qsub command.

5.	Inspect the output of the sample.
```
cat run.sh.oXXXX
```
You should see output similar to this:

```
Platform Name: Intel(R) OpenCL HD Graphics
  Platform Version: OpenCL 2.1
       Device Name: Intel(R) Gen9 HD Graphics NEO
    Max Work Group: 256
 Max Compute Units: 24

Parallel Mandelbrot set using buffers.
Rendered image output to file: mandelbrot.png (output too large to display in text)
       Serial time: 0.0430331s
     Parallel time: 0.00224131s
Successfully computed Mandelbrot set.
```

6.	Remove the stdout and stderr files and clean-up the project files.
```
rm build.sh.*; rm run.sh.*; make clean
```
7.	Disconnect from the Intel DevCloud.
```
exit
```
### Build and run additional samples
Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion to this sample. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.

