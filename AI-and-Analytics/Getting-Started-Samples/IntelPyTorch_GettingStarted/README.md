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


### Setting Environment Variables


For working at a Command-Line Interface (CLI), the tools in the oneAPI toolkits
are configured using environment variables. Set up your CLI environment by
sourcing the ``setvars`` script every time you open a new terminal window. This
will ensure that your compiler, libraries, and tools are ready for development.


#### Linux
Source the script from the installation location, which is typically in one of
these folders:


For root or sudo installations:


  ``. /opt/intel/oneapi/setvars.sh``


For normal user installations:

  ``. ~/intel/oneapi/setvars.sh``

**Note:** If you are using a non-POSIX shell, such as csh, use the following command:

     ``$ bash -c 'source <install-dir>/setvars.sh ; exec csh'``

If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


**Note:** [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html)
    can also be used to set up your development environment.
    The modulefiles scripts work with all Linux shells.


**Note:** If you wish to fine
    tune the list of components and the version of those components, use
    a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html)
    to set up your development environment.

#### Windows

Execute the  ``setvars.bat``  script from the root folder of your
oneAPI installation, which is typically:


  ``"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"``


For Windows PowerShell* users, execute this command:

  ``cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'``


If environment variables are set correctly, you will see a confirmation
message.

If you receive an error message, troubleshoot the problem using the
Diagnostics Utility for Intel® oneAPI Toolkits, which provides system
checks to find missing dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


## How to Build and Run

1. Activate conda environment With Root Access

Please follow the steps above to set up your oneAPI environment with the
`setvars.sh` script. Then navigate in Linux shell to your oneapi installation
path, typically `~/intel/inteloneapi`. Activate the conda environment with the
following command:

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