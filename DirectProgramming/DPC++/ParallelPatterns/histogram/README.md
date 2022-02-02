# `Histogram` Sample

This sample demonstrates a histogram that groups numbers together and provides the count of a particular number in the input. In this sample we are using oneDPL APIs to offload the computation to the selected device.

For comprehensive instructions regarding DPC++ Programming, go to https://software.intel.com/en-us/oneapi-programming-guide and search based on relevant terms noted in the comments.

| Optimized for                   | Description                                                                                          |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| OS                              | Linux Ubuntu 18.04                                                                                   |
| Hardware                        | Skylake with GEN9 or newer, Intel&reg; Programmable Acceleration Card with Intel&reg; Arria&reg; 10 GX FPGA|
| Software                        | Intel® oneAPI DPC++ Compiler                                                                         |


## Purpose
This sample creates both dense and sparse histograms using oneDPL APIs, on an input array of 1000 elements with values chosen randomly berween 0 and 9. To differentiate between sparse and dense histogram, we make sure that one of the values never occurs in the input array, i.e. one bin will have always have 0 value.

For the dense histogram all the bins(including the zero-size bins) are stored, whereas for the sparse algorithm only non-zero sized bins are stored.

The computations are performed using Intel® oneAPI DPC++ Library (oneDPL).

## Key Implementation Details
The basic DPC++ implementation explained in the code includes accessor,
kernels, queues, buffers as well as some oneDPL library calls.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the histogram program for CPU and GPU

### Running Samples In DevCloud
Running samples in the Intel DevCloud requires you to specify a compute node. For specific instructions, jump to [Run the Histogram sample on the DevCloud](#run-histogram-on-devcloud)


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

```
mkdir build
cd build
cmake ..
```

Build the program using the following make commands
```
make
```

Run the program using:
```
make run or src/histogram
```

Clean the program using:
```
make clean
```
If you see the following error message when compiling this sample:

```
Error 'dpc_common.hpp' file not found
```
You need to add the following directory to the list of include folders, that are required by your project, in your project's Visual Studio project property panel. The missing include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

## Running the Sample

Application Parameters
You can modify the histogram from within src/main.cpp. The functions sparse_histogram() and dense_histogram() can be reused for any set of input values.

#### Example of Output

```
Input:
1 1 8 1 8 6 1 0 1 5 5 2 2 8 1 2 1 1 1 6 2 1 1 8 3 6 6 2 2 1 1 8 1 0 0 0 2 2 7 6 5 1 6 1 1 6 1 5 1 0 0 1 1 1 0 5
 5 0 7 0 1 6 0 5 7 0 3 0 0 0 0 6 0 2 5 5 6 6 8 7 6 6 8 8 7 7 2 2 0 7 2 2 5 2 7 1 3 0 1 1 0 1 7 2 0 1 5 1 7 0 8 3 1 5 0 6 1 0 8 2 7 2 1 1 1 3 2 5 1 2 5 1 6 3 3 1 3 8 0 1 1 8 2 0 2 0 1 2 0 2 1 8 1 6 0 6 7 1 1 8 3 6 0 7 7 1 6 1 7 6 1 8 3 3 6 3 1 2 7 2 1 0 1 8 7 0 5 5 1 1 3 2 1 3 7 0 3 2 1 1 8 0 1 0 2 5 3 6 7 0 6 2 0 8 8 5 6 3 0 5 7 3 5 0 0 3 7 7 5 6 7 2 7 8 0 0 2 3 0 1 3 1 1 2 7 1 5 1 0 3 7 2 0 3 0 0 6 7 5 0 5 3 0 3 0 0 1 3 2 5 2 3 6 3 5 5 2 0 7 6 3 6 7 6 0 7 6 5 6 0 3 0 2 1 1 0 2 2 1 1 7 3 8 2 5 2 7 7 2 1 3 2 1 1 1 8 6 5 2 3 3 6 1 5 8 2 1 1 2 5 2 0 7 3 3 3 3 8 8 0 1 2 8 2 3 7 0 8 1 2 2 1 6 2 8 5 1 3 5 7 8 0 5 2 1 8 7 0 6 7 8 7 7 5 8 0 3 8 8 2 8 1 7 2 1 6 0 0 7 3 2 2 1 7 0 2 5 7 5 2 3 1 0 2 1 6 2 2 3 1 5 3 0 3 5 0 7 3 1 5 7 6 7 8 2 7 0 7 2 5 7 5 0 6 5 8 3 7 0 7 6 5 8 5 6 2 5 2 5 0 5 1 1 3 1 6 0 8 3 0 0 1 7 2 5 2 0 7 2 0 3 7 3 0 3 0 2 6 0 7 6 5 0 1 8 8 5 8 7 8 1 0 8 0 2 2 2 2 0 2 0 3 0 3 3 3 3 3 7 3 2 0 6 0 3 0 8 0 1 1 6 3 1 3 1 0 6 3 7 1 5 7 8 6 0 0 7 1 1 6 3 2 8 0 2 3 0 1 1 6 3 5 7 7 0 8 2 1 0 7 8 5 2 5 0 0 6 6 5 8 3 8 1 2 7 5 3 2 1 0 8 7 8 1 3 8 1 3 3 1 2 0 5 1 6 3 6 1 0 2 7 3 0 8 1 7 2 5 7 6 8 5 2 7 0 5 6 2 8 7 1 8 7 2 3 2 8 0 3 8 1 1 1 1 7 5 6 0 8 2 6 7 7 8 5 8 2 2 8 2 7 0 1 6 3 5 8 2 3 1 1 2 0 2 3 8 5 7 8 5 1 1 1 8 1 7 5 0 7 1 0 6 3 5 1 6 8 0 6 1 8 7 5 0 8 7 6 2 5 5 5 6 7 7 1 0 5 0 2 3 3 6 0 1 0 1 8 7 0 5 8 6 3 2 2 0 0 1 3 6 5 8 1 3 2 5 1 0 6 3 0 7 7 2 2 8 2 1 1 2 6 3 6 7 5 2 8 6 3 0 1 8 6 0 1 2 6 0 0 1 2 2 8 0 5 1 6 7 0 1 7 6 1 2 2 8 6 8 5 8 8 1 5 1 1 6 6 8 7 6 0 0 0 6 7 3 5 5 8 5 2 6 2 7 8 3 6 1 2 0 1 2 1 6 6 6 2 1 6 7 5 0 5 3 2 3 6 7 6 5 2 2 0 1 0 7 7 6 0 8 1 1 1 8 7 5 3 7 1 0 5 0 3 1 2 5 5 8 1 0 3 5 0 1 8 0 6 0 0 6 3 8 5 2 5 1 5 0 2 0 7 6 8 1 7 1 0 1 0 6 0 1 0 0 1 8 1 7 2 3 3 5 1 8 6 6 1 2 2 2 3 1 8 2 2 6 3 7 6 1 2 6 1 2 6 2 0 5 0 2 7 3 5 8 3 2 3 1 5 6 6 6 7 3 8 0 8 0 5 5 8 5 0 0 6 2 0 6 8 1 6 6 2 0 3 5 3 2 8 6 1 3 3 8 7 0 7 6 7 1 0 6 7 0 5 0 0 5 8 1
Dense Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (4, 0) (5, 105) (6, 110) (7, 108) (8, 102) ]
Sparse Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (5, 105) (6, 110) (7, 108) (8, 102) ]
```
### Running the Histogram sample in the DevCloud<a name="run-histogram-on-devcloud"></a>
1.  Open a terminal on your Linux system.
2.	Log in to DevCloud.
```
ssh devcloud
```
3.	Download the samples.
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

4. Change directories to the  Hidden Markov Model sample directory.
```
cd ~/oneAPI-samples/DirectProgramming/DPC++/ParallelPatterns/histogram
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
Input:
1 1 8 1 8 6 1 0 1 5 5 2 2 8 1 2 1 1 1 6 2 1 1 8 3 6 6 2 2 1 1 8 1 0 0 0 2 2 7 6 5 1 6 1 1 6 1 5 1 0 0 1 1 1 0 5
 5 0 7 0 1 6 0 5 7 0 3 0 0 0 0 6 0 2 5 5 6 6 8 7 6 6 8 8 7 7 2 2 0 7 2 2 5 2 7 1 3 0 1 1 0 1 7 2 0 1 5 1 7 0 8 3 1 5 0 6 1 0 8 2 7 2 1 1 1 3 2 5 1 2 5 1 6 3 3 1 3 8 0 1 1 8 2 0 2 0 1 2 0 2 1 8 1 6 0 6 7 1 1 8 3 6 0 7 7 1 6 1 7 6 1 8 3 3 6 3 1 2 7 2 1 0 1 8 7 0 5 5 1 1 3 2 1 3 7 0 3 2 1 1 8 0 1 0 2 5 3 6 7 0 6 2 0 8 8 5 6 3 0 5 7 3 5 0 0 3 7 7 5 6 7 2 7 8 0 0 2 3 0 1 3 1 1 2 7 1 5 1 0 3 7 2 0 3 0 0 6 7 5 0 5 3 0 3 0 0 1 3 2 5 2 3 6 3 5 5 2 0 7 6 3 6 7 6 0 7 6 5 6 0 3 0 2 1 1 0 2 2 1 1 7 3 8 2 5 2 7 7 2 1 3 2 1 1 1 8 6 5 2 3 3 6 1 5 8 2 1 1 2 5 2 0 7 3 3 3 3 8 8 0 1 2 8 2 3 7 0 8 1 2 2 1 6 2 8 5 1 3 5 7 8 0 5 2 1 8 7 0 6 7 8 7 7 5 8 0 3 8 8 2 8 1 7 2 1 6 0 0 7 3 2 2 1 7 0 2 5 7 5 2 3 1 0 2 1 6 2 2 3 1 5 3 0 3 5 0 7 3 1 5 7 6 7 8 2 7 0 7 2 5 7 5 0 6 5 8 3 7 0 7 6 5 8 5 6 2 5 2 5 0 5 1 1 3 1 6 0 8 3 0 0 1 7 2 5 2 0 7 2 0 3 7 3 0 3 0 2 6 0 7 6 5 0 1 8 8 5 8 7 8 1 0 8 0 2 2 2 2 0 2 0 3 0 3 3 3 3 3 7 3 2 0 6 0 3 0 8 0 1 1 6 3 1 3 1 0 6 3 7 1 5 7 8 6 0 0 7 1 1 6 3 2 8 0 2 3 0 1 1 6 3 5 7 7 0 8 2 1 0 7 8 5 2 5 0 0 6 6 5 8 3 8 1 2 7 5 3 2 1 0 8 7 8 1 3 8 1 3 3 1 2 0 5 1 6 3 6 1 0 2 7 3 0 8 1 7 2 5 7 6 8 5 2 7 0 5 6 2 8 7 1 8 7 2 3 2 8 0 3 8 1 1 1 1 7 5 6 0 8 2 6 7 7 8 5 8 2 2 8 2 7 0 1 6 3 5 8 2 3 1 1 2 0 2 3 8 5 7 8 5 1 1 1 8 1 7 5 0 7 1 0 6 3 5 1 6 8 0 6 1 8 7 5 0 8 7 6 2 5 5 5 6 7 7 1 0 5 0 2 3 3 6 0 1 0 1 8 7 0 5 8 6 3 2 2 0 0 1 3 6 5 8 1 3 2 5 1 0 6 3 0 7 7 2 2 8 2 1 1 2 6 3 6 7 5 2 8 6 3 0 1 8 6 0 1 2 6 0 0 1 2 2 8 0 5 1 6 7 0 1 7 6 1 2 2 8 6 8 5 8 8 1 5 1 1 6 6 8 7 6 0 0 0 6 7 3 5 5 8 5 2 6 2 7 8 3 6 1 2 0 1 2 1 6 6 6 2 1 6 7 5 0 5 3 2 3 6 7 6 5 2 2 0 1 0 7 7 6 0 8 1 1 1 8 7 5 3 7 1 0 5 0 3 1 2 5 5 8 1 0 3 5 0 1 8 0 6 0 0 6 3 8 5 2 5 1 5 0 2 0 7 6 8 1 7 1 0 1 0 6 0 1 0 0 1 8 1 7 2 3 3 5 1 8 6 6 1 2 2 2 3 1 8 2 2 6 3 7 6 1 2 6 1 2 6 2 0 5 0 2 7 3 5 8 3 2 3 1 5 6 6 6 7 3 8 0 8 0 5 5 8 5 0 0 6 2 0 6 8 1 6 6 2 0 3 5 3 2 8 6 1 3 3 8 7 0 7 6 7 1 0 6 7 0 5 0 0 5 8 1
Dense Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (4, 0) (5, 105) (6, 110) (7, 108) (8, 102) ]
Sparse Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (5, 105) (6, 110) (7, 108) (8, 102) ]
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
