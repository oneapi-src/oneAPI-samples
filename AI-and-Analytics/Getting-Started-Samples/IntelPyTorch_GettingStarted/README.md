# `PyTorch HelloWorld` Sample
PyTorch* is a very popular framework for deep learning. Intel and Facebook* collaborate to boost PyTorch* CPU Performance for years. The official PyTorch has been optimized using oneAPI Deep Neural Network Library (oneDNN) primitives by default. This sample demonstrates how to train a PyTorch model and shows how Intel-optimized PyTorch* enables Intel® Deep Neural Network Library (Intel® DNNL) calls by default.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Intel® Xeon® Scalable Processor family
| Software                          | Intel® oneAPI AI Analytics Toolkit
| What you will learn               | How to get started with Intel® Optimization for PyTorch
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

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

1. Activate the conda environment:

    ```
    conda activate pytorch
    ```

2. Activate conda environment Without Root Access (Optional)

By default, the Intel AI Analytics toolkit is installed in the inteloneapi
folder, which requires root privileges to manage it. If you would like to
bypass using root access to manage your conda environment, then you can clone
your desired conda environment using the following command:

    ```
    conda create --name user_pytorch --clone pytorch
    ```

    Then activate your conda environment with the following command:

    ```
    conda activate user_pytorch
    ```

4.	Navigate to the directory with the TensorFlow sample:
    ```
    cd ~/oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/IntelPyTorch_GettingStarted
    ```

5. Run the Python script
    To run the program on Linux*, Windows* and MacOS*, type the following command in the terminal with Python installed:

    ```
    python PyTorch_Hello_World.py
    ```

    You will see the DNNL verbose trace after exporting the `DNNL_VERBOSE`:

    ```
    export DNNL_VERBOSE=1
    ```

    Please find more information about the mkldnn log [here](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html).


### Example of Output
With successful execution, it will print out `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]` in the terminal.

### Running The Sample In DevCloud (Optional)

Please refer to [using samples in DevCloud](https://github.com/intel-ai-tce/oneAPI-samples/blob/devcloud/AI-and-Analytics/README.md#using-samples-in-intel-oneapi-devcloud) for general usage instructions.

### Submit The Sample in Batch Mode

1.	Navigate to the directory with the TensorFlow sample:
```
cd ~/oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/IntelPyTorch_GettingStarted
```
2. submit this "IntelPyTorch_GettingStarted" workload on the selected node with the run script.
```
./q ./run.sh
```
> the run.sh contains all the instructions needed to run this "TensorFlow_HelloWorld" workload

### Build and run additional samples
Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.