# `Intel® Extension for PyTorch (IPEX) Getting Started` Sample

Intel® Extension for PyTorch (IPEX) extends PyTorch* with optimizations for extra performance boost on Intel hardware. 

| Property             | Description
|:---                  |:---
| Category             | Get Start Sample
| What you will learn  | How to start using Intel® Extension for PyTorch (IPEX)
| Time to complete     | 15 minutes

## Purpose

This sample code demonstrates how to begin using the Intel® Extension for PyTorch (IPEX). 

The sample implements an example neural network with one convolution layer, one normalization layer, and one ReLU layer.

You can quickly build and train a PyTorch* neural network using the simple Python code. Also, by controlling the built-in environment variable, the sample attempts to show how Intel® DNNL Primitives are called explicitly and shows the performance during PyTorch* model training and inference with Intel® Extension for PyTorch (IPEX).

The Jupyter notebook in this sample also guides users how to change PyTorch* codes to run on Intel® Data Center GPU family and how to validate the GPU or CPU usages for PyTorch* workloads on Intel CPU or GPU.

>**Note**: Intel® Extension for PyTorch (IPEX) can be installed via the Intel® AI Tools Offline Installer or via the Intel AI Tools Selector. For more information on the optimizations as well as performance data, see [*Intel and Facebook* collaborate to boost PyTorch* CPU performance*](http://software.intel.com/en-us/articles/intel-and-facebook-collaborate-to-boost-pytorch-cpu-performance).

>
>Find more examples in the [*Examples*](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html) topic of the [*Intel® Extension for PyTorch (IPEX) Documentation*](https://intel.github.io/intel-extension-for-pytorch).


## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 22.04
| Hardware             | Intel® Xeon® scalable processor family <br> Intel® Data Center GPUs
| Software             | Intel® Extension for PyTorch (IPEX)

> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

## Key Implementation Details

The sample uses pretrained model provided by Intel and published as part of [Intel AI Reference Models](https://github.com/IntelAI/models). The example also illustrates how to utilize TensorFlow* and Intel® Math Kernel Library (Intel® MKL) runtime settings to maximize CPU performance on ResNet50 workload.


- The Jupyter Notebook, `ResNet50_Inference.ipynb`, is implemented for both CPU and GPU using Intel® Extension for PyTorch (IPEX).
- The `Intel_Extension_For_PyTorch_Hello_World.py` script is implemented for CPU using the Python language.
- You must export the environment variable `DNNL_VERBOSE=1` to display the deep learning primitives trace during execution.

> **Note**: The test dataset is inherited from `torch.utils.data.Dataset`, and the model is inherited from `torch.nn.Module`.

## Environment Setup
You will need to download and install the following toolkits, tools, and components to use the sample.


**1. Get Intel® AI Tools**

Required AI Tools:  Intel® Extension for PyTorch* - GPU

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
conda activate pytorch-gpu 
``` 

**4. Clone the GitHub repository**

``` 
git git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_PyTorch_GettingStarted/
```

**5. Install dependencies**

>**Note**: Before running the following commands, make sure your Conda/Python environment with AI Tools installed is activated

```
pip install -r requirements.txt
pip install notebook
``` 
For Jupyter Notebook, refer to [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.


## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](#environment-setup) is completed.

Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip) 
* [Docker](#docker)

### AI Tools Offline Installer (Validated)  

**1. Register Conda kernel to Jupyter Notebook kernel**

If the default path is used during the installation of AI Tools:
```
$HOME/intel/oneapi/intelpython/envs/pytorch-gpu/bin/python -m ipykernel install --user --name=pytorch-gpu
```
If a non-default path is used:
```
<custom_path>/bin/python -m ipykernel install --user --name=pytorch-gpu
```
**2. Launch Jupyter Notebook** 

```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```
**3. Follow the instructions to open the URL with the token in your browser**

**4. Select the Notebook**

```
ResNet50_Inference.ipynb
```

**5. Change the kernel to `pytorch-gpu`**

**6. Run every cell in the Notebook in sequence**

### Conda/PIP
> **Note**: Before running the instructions below, make sure your Conda/Python environment with AI Tools installed is activated

**1. Register Conda/Python kernel to Jupyter Notebook kernel** 

For Conda:
```
<CONDA_PATH_TO_ENV>/bin/python -m ipykernel install --user --name=pytorch-gpu
```
To know <CONDA_PATH_TO_ENV>, run `conda env list` and find your Conda environment path.

For PIP:
```
python -m ipykernel install --user --name=pytorch-gpu
```

**2. Launch Jupyter Notebook**

```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```

**3. Follow the instructions to open the URL with the token in your browser**

**4. Select the Notebook**

```
ResNet50_Inference.ipynb
```

**5. Change the kernel to `pytorch-gpu`**

**6. Run every cell in the Notebook in sequence**


### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.


## Example Output

With successful execution, it will print out `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]` in the terminal.



## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)

