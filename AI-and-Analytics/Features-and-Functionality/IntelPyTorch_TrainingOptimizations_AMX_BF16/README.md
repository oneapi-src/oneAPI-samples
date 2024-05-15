﻿# PyTorch Training Optimizations with Advanced Matrix Extensions Bfloat16 Sample

The `PyTorch Training Optimizations with Advanced Matrix Extensions Bfloat16` sample will demonstrate how to train a ResNet50 model using the CIFAR10 dataset using the Intel® Extension for PyTorch (IPEX).

The Intel® Extension for PyTorch (IPEX) extends PyTorch* with optimizations for extra performance boost on Intel® hardware. While most of the optimizations will be included in future PyTorch* releases, the extension delivers up-to-date features and optimizations for PyTorch on Intel® hardware. For example, newer optimizations include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

| Property            | Description 
|:---                 |:---
| Category            | Code Optimization
| What you will learn | How to start using Intel® Extension for PyTorch* (IPEX) with Intel® AMX BF16 for training performance improvements
| Time to complete    | 20 minutes

## Purpose

The Intel® Extension for PyTorch* (IPEX) gives users the ability to speed up training on Intel® Xeon Scalable processors with lower precision data formats and specialized computer instructions. The bfloat16 (BF16) data format uses half the bit width of floating-point-32 (FP32), lowering the amount of memory needed and execution time to process. You should notice performance optimization with the Intel® AMX instruction set when compared to AVX-512.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 22.04 or newer
| Hardware                | 4th Gen Intel® Xeon® Scalable Processors or newer
| Software                | Intel® Extension for PyTorch* (IPEX)

> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

## Key Implementation Details

- This code sample will train a ResNet50 model using the CIFAR10 dataset while using Intel® Extension for PyTorch (IPEX). The model is trained using FP32 and BF16 precision, including the use of Intel® AMX on BF16. Intel® AMX is supported on BF16 and INT8 data types starting with the 4th Generation of Xeon Scalable Processors. The training time will be compared, showcasing the speedup of BF16 and Intel® AMX.

>**Note**: Training is not performed using INT8 since using a lower precision will train a model with fewer parameters, which is likely to underfit and not generalize well.

## Environment Setup
You will need to download and install the following toolkits, tools, and components to use the sample.

**1. Get Intel® AI Tools**

Required AI Tools: Intel® Extension for PyTorch* (CPU)

If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

>**Note**: If Docker option is chosen in AI Tools Selector, refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

**2. (Offline Installer) Activate the AI Tools bundle base environment**
If the default path is used during the installation of AI Tools:
```
source $HOME/intel/oneapi/intelpython/bin/activate
```
If a non-default path is used:
```
source <custom_path>/bin/activate
```
 
**3. (Offline Installer) Activate relevant Conda environment**
```
conda activate pytorch
``` 

**4. Clone the GitHub repository**
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Features-and-Functionality/IntelPyTorch_TrainingOptimizations_AMX_BF16
```

**5. Install dependencies**
>**Note**: Before running the following commands, make sure your Conda/Python environment with AI Tools installed is activated

```
pip install -r requirements.txt
pip install notebook
``` 
For Jupyter Notebook, refer to [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.

## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPyTorch_TrainingOptimizations_AMX_BF16#environment-setup) is completed.

Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip) 
* [Docker](#docker)

### AI Tools Offline Installer (Validated)  

**1. Register Conda kernel to Jupyter Notebook kernel**

If the default path is used during the installation of AI Tools:
```
$HOME/intel/oneapi/intelpython/envs/pytorch/bin/python -m ipykernel install --user --name=pytorch
```
If a non-default path is used:
```
<custom_path>/bin/python -m ipykernel install --user --name=pytorch
```
**2. Launch Jupyter Notebook** 
```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```
**3. Follow the instructions to open the URL with the token in your browser**

**4. Select the Notebook**
```
IntelPyTorch_TrainingOptimizations_AMX_BF16.ipynb
```
**5. Change the kernel to `pytorch`**

**6. Run every cell in the Notebook in sequence**

### Conda/PIP
> **Note**: Before running the instructions below, make sure your Conda/Python environment with AI Tools installed is activated

**1. Register Conda/Python kernel to Jupyter Notebook kernel** 
For Conda:
```
<CONDA_PATH_TO_ENV>/bin/python -m ipykernel install --user --name=<your-env-name>
```
To know <CONDA_PATH_TO_ENV>, run `conda env list` and find your Conda environment path.

For PIP:
```
python -m ipykernel install --user --name=<your-env-name>
```
**2. Launch Jupyter Notebook**
```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```
**3. Follow the instructions to open the URL with the token in your browser**

**4. Select the Notebook**
```
IntelPyTorch_TrainingOptimizations_AMX_BF16.ipynb
```
**5. Change the kernel to `<your-env-name>`**

**6. Run every cell in the Notebook in sequence**

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample will print out the runtimes and charts of relative performance with the FP32 model without any optimizations as the baseline. 

The performance speedups using Intel® AMX BF16 are approximate on ResNet50. Performance will vary based on your hardware and software versions. To see more performance improvement between AVX-512 BF16 and Intel® AMX BF16, increase the batch size with CIFAR10 or use another dataset. For even more speedup, consider using the Intel® Extension for PyTorch (IPEX) [Launch Script](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/launch_script.html).  

## Related Samples

* [PyTorch* Inference Optimizations with Advanced Matrix Extensions Bfloat16 Integer8](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPyTorch_InferenceOptimizations_AMX_BF16_INT8)
* [Intel PyTorch GPU Inference Optimization with AMP](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPyTorch_GPU_InferenceOptimization_with_AMP)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
