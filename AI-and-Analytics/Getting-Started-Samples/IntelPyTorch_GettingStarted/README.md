# `PyTorch HelloWorld` Sample
PyTorch* is a very popular framework for deep learning. Intel and Facebook* collaborate to boost PyTorch* CPU Performance for years. The official PyTorch has been optimized using oneAPI Deep Neural Network Library (oneDNN) primitives by default. This sample demonstrates how to train a PyTorch model and shows how Intel-optimized PyTorch* enables Intel® DNNL calls by default. 

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Intel® Xeon® Scalable Processor family
| Software                          | Intel&reg; oneAPI AI Analytics Toolkit
| What you will learn               | How to get started with Intel Optimization for PyTorch
| Time to complete                  | 15 minutes

## Purpose
This sample code shows how to get started with Intel Optimization for PyTorch. It implements an example neural network with one convolution layer, one normalization layer and one ReLU layer. Developers can quickly build and train a PyTorch* neural network using a simple python code. Also, by controlling the build-in environment variable, the sample attempts to show how Intel® DNNL Primitives are called explicitly and their performance during PyTorch* model training and inference.

Intel-optimized PyTorch* is available as part of Intel® AI Analytics Toolkit. For more information on the optimizations as well as performance data, see this blog post http://software.intel.com/en-us/articles/intel-and-facebook-collaborate-to-boost-pytorch-cpu-performance.

## Key implementation details
This Hello World sample code is implemented for CPU using the Python language. 

*Please* **export the environment variable `DNNL_VERBOSE=1`** *to display the deep learning primitives trace during execution.*

### Notes
 - The test dataset is inherited from `torch.utils.data.Dataset`.
 - The model is inherited from `torch.nn.Module`.
 - For the inference portion, `to_mkldnn()` function in `torch.utils.mkldnn` can accelerate performance by eliminating data reorders between operations, which are supported by Intel&reg; DNNL.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## How to Build and Run
### Running Samples In DevCloud (Optional)

<!---Include the next paragraph ONLY if the sample runs in batch mode-->
### Run in Batch Mode
This sample runs in batch mode, so you must have a script for batch processing. Once you have a script set up, refer to the [Tensorflow Hello World](https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/Getting-Started-Samples/IntelTensorFlow_GettingStarted/README.md) instructions or the [PyTorch Hello World](https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/Getting-Started-Samples/IntelPyTorch_GettingStarted/README.md) instructions to run the sample.

<!---Include the next paragraph ONLY if the sample DOES NOT RUN in batch mode-->
### Run in Interactive Mode
This sample runs in interactive mode. Follow the directions in the README.md for the sample you want to run. If the sample can be run in interactive mode, the sample will have directions on how to run the sample in a Jupyter Notebook. An example can be found in the [Intel&reg; Modin Getting Started](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/IntelModin_GettingStarted) sample.

### Request a Compute Node
In order to run on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
 This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

<!---Mark each compatible Node in BOLD-->
| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
| CPU               | qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh          |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |



1. Pre-requirement

    PyTorch is ready for use once you finish the Intel&reg; AI Analytics Toolkit installation and have run the post installation script. These steps apply to DevCloud as well.

    You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

2. Activate conda environment With Root Access

    Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the setvars.sh script. Then navigate in Linux shell to your oneapi installation path, typically `~/intel/inteloneapi`. Activate the conda environment with the following command:

    ```
    conda activate pytorch
    ```

3. Activate conda environment Without Root Access (Optional)

    By default, the Intel AI Analytics toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

    ```
    conda create --name user_pytorch --clone pytorch
    ```

    Then activate your conda environment with the following command:

    ```
    conda activate user_pytorch
    ```

4. Run the Python script
    To run the program on Linux*, Windows* and MacOS*, type the following command in the terminal with Python installed:

    ```
    python PyTorch_Hello_World.py
    ```

    You will see the DNNL verbose trace after exporting the `DNNL_VERBOSE`:

    ```
    export DNNL_VERBOSE=1
    ```

    Please find more information about the mkldnn log [here](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html).

## Example of Output
With successful execution, it will print out `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]` in the terminal.
