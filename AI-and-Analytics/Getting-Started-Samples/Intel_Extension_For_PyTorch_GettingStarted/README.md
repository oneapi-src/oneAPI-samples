# `Intel® Extension for PyTorch* Getting Started` Sample

Intel® Extension for PyTorch* extends PyTorch* with optimizations for extra performance boost on Intel hardware. Most of the optimizations will be included in stock PyTorch* releases eventually, and the intention of the extension is to deliver up-to-date features and optimizations for PyTorch* on Intel hardware, examples include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

This sample contains a Jupyter* NoteBook that guides you through the process of running a PyTorch* inference workload on both GPU and CPU by using Intel® AI Analytics Toolkit (AI Kit) and also analyze the GPU and CPU usage via Intel® oneAPI Deep Neural Network Library (oneDNN) verbose logs.

| Area                 | Description
|:---                  |:---
| What you will learn  | How to get started with Intel® Extension for PyTorch
| Time to complete     | 15 minutes

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 22.04
| Hardware             | Intel® Xeon® scalable processor family <br> Intel® Data Center GPUs
| Software             | Intel® AI Analytics Toolkit (AI Kit)


## Hardware requirement

Verified Hardware Platforms for CPU samples:
 - Intel® CPU (Xeon, Core)

Verified Hardware Platforms for GPU samples:
 - [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html)
 - [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/docs/processors/max-series/overview.html)
 - [Intel® Arc™ Graphics](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) (experimental)

## Purpose

This sample code demonstrates how to begin using the Intel® Extension for PyTorch*. 

The sample implements an example neural network with one convolution layer, one normalization layer, and one ReLU layer.

You can quickly build and train a PyTorch* neural network using the simple Python code. Also, by controlling the built-in environment variable, the sample attempts to show how Intel® DNNL Primitives are called explicitly and shows the performance during PyTorch* model training and inference with Intel® Extension for PyTorch*.

The Jupyter notebook in this sample also guides users how to change PyTorch* codes to run on Intel® Data Center GPU family and how to validate the GPU or CPU usages for PyTorch* workloads on Intel CPU or GPU.

>**Note**: Intel® Extension for PyTorch* is available as part of Intel® AI Analytics Toolkit. For more information on the optimizations as well as performance data, see [*Intel and Facebook* collaborate to boost PyTorch* CPU performance*](http://software.intel.com/en-us/articles/intel-and-facebook-collaborate-to-boost-pytorch-cpu-performance).
>
>Find more examples in the [*Examples*](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html) topic of the [*Intel® Extension for PyTorch* Documentation*](https://intel.github.io/intel-extension-for-pytorch).


## Key Implementation Details

The sample uses pretrained model provided by Intel and published as part of [Intel Model Zoo](https://github.com/IntelAI/models). The example also illustrates how to utilize TensorFlow* and Intel® Math Kernel Library (Intel® MKL) runtime settings to maximize CPU performance on ResNet50 workload.

- The Jupyter Notebook, `ResNet50_Inference.ipynb`, is implemented for both CPU and GPU using Intel® Extension for PyTorch*.
- The `Intel_Extension_For_PyTorch_Hello_World.py` script is implemented for CPU using the Python language.
- You must export the environment variable `DNNL_VERBOSE=1` to display the deep learning primitives trace during execution.

> **Note**: The test dataset is inherited from `torch.utils.data.Dataset`, and the model is inherited from `torch.nn.Module`.

## Run the `Intel® Extension for PyTorch* Getting Started` Sample

### On Linux*

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

#### Activate Conda

1. Activate the conda environment:
   ```
   conda activate pytorch
   ```

2. Activate conda environment without Root access (Optional).

   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.
   ```
   conda create --name user_pytorch --clone pytorch
   ```
   Then activate your conda environment with the following command:
   ```
   conda activate user_pytorch
   ```
#### Run the Script

1.	Navigate to the directory with the sample.
    ```
    cd ~/oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_PyTorch_GettingStarted
    ```
2. Run the Python script.
   ```
   python Intel_Extension_For_PyTorch_Hello_World.py
   ```
   You will see the DNNL verbose trace after exporting the `DNNL_VERBOSE`:
   ```
   export DNNL_VERBOSE=1
   ```
   >**Note**: Read more information about the mkldnn log at [https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html).

#### Run the Jupyter Notebook

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
   ```
   ResNet50_Inference.ipynb
   ```
5. Change your Jupyter Notebook kernel to **PyTorch**.
6. Run every cell in the Notebook in sequence.

### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

### Run the `Intel® Extension for PyTorch* Getting Started` Sample on Intel® DevCloud (Optional)

1. If you do not already have an account, request an Intel® DevCloud account at [*Create an Intel® DevCloud Account*](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).
2. On a Linux* system, open a terminal.
3. SSH into Intel® DevCloud.
   ```
   ssh DevCloud
   ```
   > **Note**: You can find information about configuring your Linux system and connecting to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://DevCloud.intel.com/oneapi/get_started).

  
4. Navigate to the directory with the sample.
   ```
   cd ~/oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_PyTorch_GettingStarted
   ```
5. Submit this `Intel_Extension_For_PyTorch_GettingStarted` workload on the selected node with  the run script.
   ```
   ./q ./run.sh
   ```
   The `run.sh` script contains all the instructions needed to run this `Intel_Extension_For_PyTorch_Hello_World.py` workload.

### Example Output

With successful execution, it will print out `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]` in the terminal.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
