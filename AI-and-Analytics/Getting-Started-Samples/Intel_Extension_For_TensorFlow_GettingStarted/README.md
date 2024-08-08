# `Intel® Extension for TensorFlow* (ITEX) Getting Started` Sample

This code sample will guide users how to run a TensorFlow* inference workload on both GPU and CPU by using Intel® AI Tools and also analyze the GPU and CPU usage via oneDNN verbose logs.

| Property             | Description
|:---                  |:---
| Category             | Get Started Sample
| What you will learn  | How to start using Intel® Extension for TensorFlow* (ITEX)
| Time to complete     | 15 minutes

## Purpose
  - Guide users how to use different conda environments in Intel® AI Tools to run TensorFlow* workloads on both CPU and GPU.
  - Guide users how to validate the GPU or CPU usages for TensorFlow* workloads on Intel CPU or GPU, using ResNet50v1.5 as an example.



## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 22.04
| Hardware             | Intel® Xeon® scalable processor family <br> Intel® Data Center GPU Max Series <br> Intel® Data Center GPU Flex Series <br> Intel® Arc™ A-Series |
| Software             | Intel® Extension for TensorFlow* (ITEX)

> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).


## Key implementation details
1. leverage the [resnet50 inference sample](https://github.com/intel/intel-extension-for-tensorflow/tree/main/examples/infer_resnet50) from intel-extension-for-tensorflow
2. use the resnet50v1.5 pretrained model from TensorFlow Hub
3. infernece with images in intel caffe github
4. guide users how to use different conda environment to run on Intel CPU and GPU
5. analyze oneDNN verbose logs to validate GPU or CPU usage  


## Environment Setup
You will need to download and install the following toolkits, tools, and components to use the sample.

**1. Get Intel® AI Tools**

Required AI Tools: `Intel® Extension for TensorFlow*`
<br>If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

**2. Install dependencies**
```
pip install -r requirements.txt
```
**Install Jupyter Notebook** by running `pip install notebook`. Alternatively, see [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.

## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](#environment-setup) is completed.
Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip) 
* [Docker](#docker)

### AI Tools Offline Installer (Validated)  
1. If you have not already done so, activate the AI Tools bundle base environment. If you used the default location to install AI Tools, open a terminal and type the following
```
source $HOME/intel/oneapi/intelpython/bin/activate
```
If you used a separate location, open a terminal and type the following
```
source <custom_path>/bin/activate
```
2. Activate the Conda environment:
```
conda activate tensorflow-gpu ## For the system with Intel GPU
conda activate tensorflow ## For the system with Intel CPU  
``` 
3. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_TensorFlow_GettingStarted
```
4. Launch Jupyter Notebook: 
> **Note**: You might need to register Conda kernel to Jupyter Notebook kernel, 
feel free to check [the instruction](https://github.com/IntelAI/models/tree/master/docs/notebooks/perf_analysis#option-1-conda-environment-creation)
```
jupyter notebook --ip=0.0.0.0
```
5. Follow the instructions to open the URL with the token in your browser.
6. Select the Notebook:
```
ResNet50_Inference.ipynb
```
7. Change the kernel to `tensorflow-gpu` for system with Intel GPU or to `tensorflow` for system with Intel CPU.
8. Run every cell in the Notebook in sequence.

### Conda/PIP
> **Note**: Make sure your Conda/Python environment with AI Tools installed is activated
1. Clone the GitHub repository:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneapi-samples/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_TensorFlow_GettingStarted
```
2. Launch Jupyter Notebook: 
> **Note**: You might need to register Conda kernel to Jupyter Notebook kernel, 
feel free to check [the instruction](https://github.com/IntelAI/models/tree/master/docs/notebooks/perf_analysis#option-1-conda-environment-creation)
```
jupyter notebook --ip=0.0.0.0
```
4. Follow the instructions to open the URL with the token in your browser.
5. Select the Notebook:
```
ResNet50_Inference.ipynb
```
7. Change the kernel to `tensorflow-gpu` for system with Intel GPU or to `tensorflow` for system with Intel CPU.
8. Run every cell in the Notebook in sequence.

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/2024-0/overview.html)


## Example Output
With successful execution, it will print out `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]` in the terminal.


## Related Samples

Find more examples in [Intel® Extension for TensorFlow* (ITEX) examples documentation]([https://intel.github.io/intel-extension-for-tensorflow/latest/examples/README.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
