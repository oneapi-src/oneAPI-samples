# `PyTorch Training Optimizations with Advanced Matrix Extensions Bfloat16` Sample

The `PyTorch Training Optimizations with Advanced Matrix Extensions Bfloat16` sample will demonstrate how to train a ResNet50 model using the CIFAR10 dataset using the Intel® Extension for PyTorch (IPEX).

The Intel® Extension for PyTorch (IPEX) extends PyTorch* with optimizations for extra performance boost on Intel® hardware. While most of the optimizations will be included in future PyTorch* releases, the extension delivers up-to-date features and optimizations for PyTorch on Intel® hardware. For example, newer optimizations include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

| Area                  | Description
|:---                   |:---
| What you will learn   | Training performance improvements using Intel® Extension for PyTorch (IPEX) with Intel® AMX BF16
| Time to complete      | 20 minutes
| Category              | Code Optimization

## Purpose

The Intel® Extension for PyTorch (IPEX) gives users the ability to speed up training on Intel® Xeon Scalable processors with lower precision data formats and specialized computer instructions. The bfloat16 (BF16) data format uses half the bit width of floating-point-32 (FP32), lowering the amount of memory needed and execution time to process. You should notice performance optimization with the Intel® AMX instruction set when compared to AVX-512.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04 or newer
| Hardware                | 4th Gen Intel® Xeon® Scalable Processors or newer
| Software                | Intel® Extension for PyTorch (IPEX)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html).

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See *[Intel® DevCloud for oneAPI](https://DevCloud.intel.com/oneapi/get_started/)* for information.

## Key Implementation Details

This code sample will train a ResNet50 model using the CIFAR10 dataset while using Intel® Extension for PyTorch (IPEX). The model is trained using FP32 and BF16 precision, including the use of Intel® AMX on BF16. Intel® AMX is supported on BF16 and INT8 data types starting with the 4th Generation of Xeon Scalable Processors. The training time will be compared, showcasing the speedup of BF16 and Intel® AMX.

>**Note**: Training is not performed using INT8 since using a lower precision will train a model with fewer parameters, which is likely to underfit and not generalize well.

The sample tutorial contains one Jupyter Notebook and a Python script. You can use either.

### Jupyter Notebook

| Notebook                              | Description
|:---                                   |:---
|`IntelPyTorch_TrainingOptimizations_AMX_BF16.ipynb` | PyTorch Training Optimizations with Advanced Matrix Extensions Bfloat16

### Python Scripts

| Script                             | Description
|:---                                |:---
|`pytorch_training_amx_bf16.py`      | The script performs training with Intel® AMX BF16 and compares the performance against the baseline
|`pytorch_training_avx512_bf16.py`   | The script performs training with AVX512 in BF16

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `PyTorch Training Optimizations with Advanced Matrix Extensions Bfloat16` Sample

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
    conda activate pytorch
    ```
2. Activate Conda environment without Root access (Optional).

   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

   ```
   conda create --name user_pytorch --clone pytorch
   conda activate user_pytorch
   ```

#### Additional Environment Setup

- **Additional Packages**

  You will need to install these additional packages in *requirements.txt*.
  ```
  python -m pip install -r requirements.txt
  ```

- **Jupyter Kernelspec**

  Add the jupyter kernelspec. This step is essential to ensure the notebook uses the environment you set up.
  ```
  python -m ipykernel install --user --name=user_pytorch
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
   IntelPyTorch_TrainingOptimizations_AMX_BF16.ipynb
   ```
5. Change your Jupyter Notebook kernel to **user_pytorch**.
6. Run every cell in the Notebook in sequence.

#### Running on the Command Line (Optional)

1. Change to the sample directory.
2. Run the script.
   ```
   python pytorch_training_amx_bf16.py
   python pytorch_training_avx512_bf16.py
   ```

### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample will print out the runtimes and charts of relative performance with the FP32 model without any optimizations as the baseline. 

The performance speedups using Intel® AMX BF16 are approximate on ResNet50. Performance will vary based on your hardware and software versions. To see more performance improvement between AVX-512 BF16 and Intel® AMX BF16, increase the batch size with CIFAR10 or use another dataset. For even more speedup, consider using the Intel® Extension for PyTorch (IPEX) [Launch Script](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/launch_script.html). 

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)