# `Matrix Multiply` Sample
A sample containing multiple implementations of matrix multiplication code sample and  is implemented using the DPC++ language for CPU and GPU.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler; Intel&reg; Advisor
| What you will learn               | How to profile an application using Intel&reg; Advisor
| Time to complete                  | 15 minutes

## Purpose

The Matrix Multiplication sample performs basic matrix multiplication. Three versions are provided that use different features of DPC++.

## Key Implementation details

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


### Running Samples In DevCloud
Running samples in the Intel DevCloud requires you to specify a compute node. For specific instructions, jump to [Run the Matrix Multiply Advisor sample on the DevCloud](#run-matmul-advisor-on-devcloud)

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

## How to Build

This sample contains 3 version of matrix multiplication using DPC++:

    multiply1 – basic implementation of matrix multiply using DPC++
    multiply1_1 – basic implementation that replaces the buffer store with a local accessor “acc” to reduce memory traffic
    multiply1_2 – the basic implementation, plus adding the local accessor and matrix tiling

Edit the line in src/multiply.hpp to select the version of the multiply function:
#define MULTIPLY multiply1


### On a Linux* System
	To build DPC++ version:
	cd <sample dir>
	cmake .
	make

    Clean the program
    make clean

### On a Windows* System Using Visual Studio 2017 or newer
   * Open Visual Studio 2017
   * Select Menu "File > Open > Project/Solution", find "matrix_multiply" folder and select "matrix_multiply.sln"
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program

### on Windows - command line - Build the program using MSBuild
    DPCPP Configurations:
    Release - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Release"
    Debug - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Debug"



### Example of Output

   ./matrix.dpcpp

   Using multiply kernel: multiply1

   Running on Intel(R) Gen9

   Elapsed Time: 0.539631s


## Running an Intel Advisor analysis
------------------------------------------

See the Advisor Cookbook here: https://software.intel.com/en-us/advisor-cookbook


### Running the Matrix Multiply Advisor sample in the DevCloud<a name="run-matmul-advisor-on-devcloud"></a>
This sample contains 3 version of matrix multiplication using DPC++:

    multiply1 – basic implementation of matrix multiply using DPC++
    multiply1_1 – basic implementation that replaces the buffer store with a local accessor “acc” to reduce memory traffic
    multiply1_2 – the basic implementation, plus adding the local accessor and matrix tiling

Edit the line in src/multiply.hpp to select the version of the multiply function:
#define MULTIPLY multiply1

1.  Open a terminal on your Linux system.
2.	Log in to DevCloud.
```
ssh devcloud
```
3.	Download the samples.
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

4. Change directories to the Matrix Multiply Advisor sample directory.
```
cd ~/oneAPI-samples/Tools/Advisor/matrix_multiply_advisor
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
Note: The watch -n 1 command is used to run qstat -n -1 and display its results every second. If no results are displayed, the job has completed.

3.	After the build job completes successfully, run the sample on a gpu node:
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
Scanning dependencies of target run
Address of buf1 = 0x7f570456f010
Offset of buf1 = 0x7f570456f180
Address of buf2 = 0x7f5703d6e010
Offset of buf2 = 0x7f5703d6e1c0
Address of buf3 = 0x7f570356d010
Offset of buf3 = 0x7f570356d100
Address of buf4 = 0x7f5702d6c010
Offset of buf4 = 0x7f5702d6c140
Using multiply kernel: multiply1
Running on Intel(R) UHD Graphics P630 [0x3e96]
Elapsed Time: 1.79388s
Built target run
```

6.	Remove the stdout and stderr files and clean-up the project files.
```
rm build.sh.*; rm run.sh.*; make clean
```
7.	Disconnect from the Intel DevCloud.
```
exit
```
## Running an Intel Advisor analysis
------------------------------------------

See the Advisor Cookbook here: https://software.intel.com/en-us/advisor-cookbook

### Build and run additional samples
Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion to this sample. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.

