# DPC++ Hidden Markov Model` Sample
The HMM (Hidden Markov Model) sample presents a statistical model using a Markov process to present graphable nodes that are otherwise in an unobservable state or “hidden”.  This technique helps with pattern recognition such as speech, handwriting, gesture recognition, part-of-speech tagging, partial discharges and bioinformatics. The sample offloads the complexity of the Markov process to the GPU.

The directed edges of this graph are possible transitions between nodes or states defined with the following parameters:
- the number of states is N, the transition matrix A is a square matrix of size N.
- Each element with indexes (i,j) of this matrix determines the probability to move from the state i to the state j on any step of the Markov process (i and j can be the same if the state does not change on the taken step).

HMM's main assumption is that there are visible observations that depend on the current Markov process. That dependency can be described as a conditional probability distribution (represented by emission matrix). The problem is to find out the most likely chain of the hidden Markov states using the given observations set.

## Requirements and sample info

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer,
| Software                          | Intel&reg; oneAPI DPC++ Compiler
| What you will learn               | Implement Viterbi algorithm to get the most likely path that consists of the hidden states
| Time to complete                  | 1 minute

## Purpose

The sample can use GPU offload to compute sequential steps of multiple graph traversals simultaneously.

This code sample implements the Viterbi algorithm, a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in Markov information sources and HMM.

- Initially, the dataset for algorithm processing is generated: initial states probability distribution Pi, transition matrix A, emission matrix B and the sequence or the observations produced by hidden Markov process.
- First, the matrix of Viterbi values on the first states are initialized using distribution Pi and emission matrix B. The matrix of back pointers is initialized with default values -1.
- Then, for each time step, the Viterbi matrix is set to the maximal possible value using A, B and Pi.
- Finally, the state with maximum Viterbi value on the last step is set as a Viterbi path's final state. The previous nodes of this path are determined using the back pointers matrix's correspondent rows for each step except the last one.

Note: The implementation uses logarithms of the probabilities to process small numbers correctly and replace multiplication operations with addition operations.

## Key Implementation details

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the `DPC++ Hidden Markov Model` Program for CPU and GPU

### Running Samples In DevCloud
Running samples in the Intel DevCloud requires you to specify a compute node. For specific instructions, jump to [Run the Hidden Markov Model sample in the DevCloud](#run-hmm-on-devcloud)


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

### Include Files
The include folder is located at %ONEAPI_ROOT%\dev-utilities\latest\include on your development system.

### On a Linux* System
1. Build the program using the following `cmake` commands.
    ```
    $ cd hidden-markov-models
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```

2. Run the program:
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```

### On a Windows* System Using a Command Line Interface
* Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

* Build the program using MSBuild
    - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
    - Run the following command: `MSBuild hidden-markov-models.sln /t:Rebuild /p:Configuration="Release"`

### On a Windows* System Using Visual Studio* Version 2017 or Newer
Perform the following steps:
1. Locate and select the `hidden-markov-models.sln` file.
2. Select the configuration 'Debug' or 'Release'.
3. Select **Project** > **Build** menu option to build the selected configuration.
4. Select **Debug** > **Start Without Debugging** menu option to run the program.

## Running the Sample

### Application Parameters
There are no editable parameters for this sample.

### Example of Output

```
Device: Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz Intel(R) OpenCL
The Viterbi path is:
19 18 17 16 15 14 13 12 11 10
The sample completed successfully!
```

### Running the Hidden Markov Model sample in the DevCloud<a name="run-hmm-on-devcloud"></a>
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
cd ~/oneAPI-samples/DirectProgramming/DPC++/GraphTraversal/hidden-markov-models
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
[100%] Built target hidden-markov-models
Scanning dependencies of target run
Device: Intel(R) UHD Graphics P630 [0x3e96] Intel(R) Level-Zero
The Viterbi path is:
16 4 17 0 16 8 16 4 17 0 1 4 17 8 16 8 16 8 12 11
The sample completed successfully!
[100%] Built target run
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
Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion to iso3dfd_omp_offload. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.
