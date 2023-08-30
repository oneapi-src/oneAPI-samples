# `Intel PyTorch GPU Inference Optimization with AMP` Sample

The `Intel PyTorch GPU Inference Optimization with AMP` sample will demonstrate how to use PyTorch ResNet50 model transfer learning and inference using the CIFAR10 dataset on Intel discrete GPU with Intel® Extension for PyTorch*.

Intel® Extension for PyTorch* extends PyTorch* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* XPU device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

| Area                  | Description
|:---                   |:---
| What you will learn   | Training with FP32 and AMP BF16 and Inference improvements with AMP BF16 on GPU using Intel® Extension for PyTorch*
| Time to complete      | 20 minutes
| Category              | Code Optimization

## Purpose

The Intel® Extension for PyTorch* gives users the ability to perform PyTorch model training and inference on Intel® discrete GPUs. It also supports lower-precision data formats and specialized computer instructions. The bfloat16(BF16) data format uses half the bit width of floating-point-32 (FP32), lowering the amount of memory needed and execution time to process. Due to its special nature of having the same number of bits for exponent as FP32, BF16 can easily be converted to FP32 and Vice Versa. This allows Auto-Mixed Precision(AMP) Training and Inference easy with FP32 and BF16. To support and accelerate BF16 data type, Intel discrete GPUs have Intel® Xe Matrix Extensions(XMX) instructions, with that you should notice performance optimization over FP32.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 22.04 or newer
| Hardware                | Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series, and Intel® ARC™ A-Series GPUs(Experimental Support)
| Software                | Intel® oneAPI AI Analytics Toolkit 2023.1 or later

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit) 2023.1 or later**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

- **Additional Packages**

  You will need to install these additional packages, already added in requirements.txt file: **Matplotlib**, **requests**, **tqdm**
  ```
  pip install -r requirements.txt
  ```


## Key Implementation Details

This code sample will train a ResNet50 model using the CIFAR10 dataset while using Intel® Extension for PyTorch*. The model is trained using FP32 by default but can also be trained with AMP BF16 precision by passing BF16 parameter in the Train function. Then the same trained model is taken and inference with FP32 and AMP BF16 is done and latency is compared to see the performance improvement with the use of Intel® Xe Matrix Extensions(XMX) for BF16. XMX is supported on BF16 and INT8 data types on Intel discrete GPUs.

>**Note**: Training is not performed using INT8 since a lower precision will train a model with fewer parameters, which is likely to underfit and not generalize well.

The sample tutorial contains one Jupyter Notebook and a Python script. You can use either.

### Jupyter Notebook

| Notebook                                               | Description
|:---                                                    |:---
|`IntelPyTorch_GPU_InferenceOptimization_with_AMP.ipynb` | PyTorch AMP BF16 Training+Inference

### Python Scripts

| Script                                              | Description
|:---                                                 |:---
|`IntelPyTorch_GPU_InferenceOptimization_with_AMP.py` | The script performs training with FP32 & inference with FP32 and AMP BF16 & compares the inference performance of AMP BF16 with FP32.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Intel PyTorch GPU Inference Optimization with AMP` Sample

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

1. Activate the Conda environment.
    ```
    conda activate pytorch-gpu
    ```
2. Activate Conda environment without Root access (Optional).

   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment and create a jupyter kernal using the following commands similar to the following.

   ```
   conda create --name user_pytorch-gpu --clone pytorch-gpu
   conda activate user_pytorch-gpu
   python -m ipykernel install --user --name=PyTorch-GPU
   ```

#### Running the Jupyter Notebook

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
   ```
   IntelPyTorch_GPU_InferenceOptimization_with_AMP.ipynb
   ```
5. Change your Jupyter Notebook kernel to **PyTorch (AI kit)**.
6. Run every cell in the Notebook in sequence.

#### Running on the Command Line (Optional)

1. Change to the sample directory.
2. Run the script.
   ```
   python IntelPyTorch_GPU_InferenceOptimization_with_AMP.py
   ```


### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample generates performance and analysis diagrams for comparison.

The following image shows approximate performance speed increases using AMX BF16 with auto-mixed precision during training.



## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
