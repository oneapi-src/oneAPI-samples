#`DPC++ Hidden Markov Model` Sample
HMM (Hidden Markov Model) is the statistical model in which the system is represented by a Markov process with the unobservable or so called "hidden" states. 
States can be represented as nodes of directed graph. The directed edges of this graph are possible transitions beetween nodes defined with some probabilities.
The number of states is N, the transition matrix A is a square matrix of size N. Each element with indexes (i,j) of this matrix determines the probability to move from 
the state i to the state j on any step of the Markov process (i and j can be the same if the state does not change on the taken step).
The main assumption of the HMM is that there are visible observations that depend on the current Markov process. 
That dependency can be described as a conditional probability distribution. The problem is to find out the chain of the hidden Markov states using the given observations set. 

##Requirements and sample info

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer,
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | Implement Viterbi algorithm to get the most probable path that consists of the hidden states 

##Purpose

This code sample implements the Viterbi algorithm which is a dynamic programming algorithm for finding
the most likely sequence of hidden states—called the Viterbi path—that results in a sequence
of observed events, especially in the context of Markov information sources and HMM.

##Key Implementation details


## License  
This code sample is licensed under MIT license. 

## Building the `DPC++ Hidden Markov Model` Program for CPU and GPU 

### On a Linux* System


### On a Windows* System Using a Command Line Interface


### On a Windows* System Using Visual Studio* Version 2017 or Newer
Perform the following steps:
1. Launch the Visual Studio* 2017.
2. Select the menu sequence **File** > **Open** > **Project/Solution**. 
3. Locate the `hidden-markov-models` folder.
4. Select the `hidden-markov-models.sln` file.
5. Select the configuration 'Debug' or 'Release'  
6. Select **Project** > **Build** menu option to build the selected configuration.
7. Select **Debug** > **Start Without Debugging** menu option to run the program.

## Running the Sample
### Application Parameters
There are no editable parameters for this sample.

### Example of Output
<pre>
Running on device:        Intel(R) Gen9 HD Graphics NEO

Successfully completed on device.</pre>