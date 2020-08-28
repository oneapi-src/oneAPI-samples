#`DPC++ Hidden Markov Model` Sample
The HMM (Hidden Markov Model) sample presents a statistical model using a Markov process to present graphable nodes that are otherwise in an unobservable state or “hidden”.  This technique is helpful in pattern recognition such as speech, handwriting, gesture recognition, part-of-speech tagging, partial discharges and bioinformatics. The sample offloads the complexity of the Markov process to the GPU.

The directed edges of this graph are possible transitions beetween nodes or states defined with the following parameters: the number of states is N, the transition matrix A is a square matrix of size N. Each element with indexes (i,j) of this matrix determines the probability to move from the state i to the state j on any step of the Markov process (i and j can be the same if the state does not change on the taken step).

The main assumption of the HMM is that there are visible observations that depend on the current Markov process. That dependency can be described as a conditional probability distribution (represented by emission matrix). The problem is to find out the most likely chain of the hidden Markov states using the given observations set.

##Requirements and sample info

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer,
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | Implement Viterbi algorithm to get the most likely path that consists of the hidden states
| Time to complete                  | 1 minute

##Purpose

The sample can use GPU offload to compute sequential steps of multiple graph traversals simultaneously.

This code sample implements the Viterbi algorithm which is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in the context of Markov information sources and HMM.

- Initially, the dataset for algorithm processing is generated: initial states probability distribution Pi, transition matrix A, emission matrix B and the sequence or the observations produced by hidden Markov process.
- First, the matrix of Viterbi values on the first states are initialized using distribution Pi and emission matrix B. The matrix of back pointers is initialized with default values -1.
- Then, for each time step the Viterbi matrix is set to the maximal possible value using A, B and Pi.
- Finally, the state with maximum Viterbi value on the last step is set as a final state of the Viterbi path and the previous nodes of this path are detemined using the correspondent rows of back pointers matrix for each of the steps except the last one.

Note: The implementation uses logarithms of the probabilities to process small numbers correctly and to replace multiplication operations with addition operations.

##Key Implementation details

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## License
This code sample is licensed under MIT license.

## Building the `DPC++ Hidden Markov Model` Program for CPU and GPU

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
      Right click on the solution file and open using either VS2017 or VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.

    * Build the program using MSBuild
      Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for
 VS2019"
      Run - MSBuild hidden-markov-models.sln /t:Rebuild /p:Configuration="Release"

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
Device: Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz Intel(R) OpenCL
The Viterbi path is:
19 18 17 16 15 14 13 12 11 10
The sample completed successfully!