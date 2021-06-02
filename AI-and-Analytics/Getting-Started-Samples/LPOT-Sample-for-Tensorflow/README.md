# `Intel® Low Precision Optimization Tool (LPOT)` Sample for TensorFlow*

## Background
Low-precision inference can speed up inference obviously, by converting the fp32 model to int8 or bf16 model. Intel provides Intel&reg; Deep Learning Boost technology in the Second Generation Intel&reg; Xeon&reg; Scalable Processors and newer Xeon&reg;, which supports to speed up int8 and bf16 model by hardware.

Intel&reg; Low Precision Optimization Tool (LPOT) helps the user to simplify the processing to convert the fp32 model to int8/bf16.

At the same time, LPOT will tune the quanization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

LPOT is released in Intel&reg; AI Analytics Toolkit and works with Intel&reg; Optimization of TensorFlow*.

Please refer to the official website for detailed info and news: [https://github.com/intel/lp-opt-tool](https://github.com/intel/lp-opt-tool)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Purpose
This sample will show a whole process to build up a CNN model to recognize handwriting number and speed up it by LPOT.

We will learn how to train a CNN model based on Keras with TensorFlow, use LPOT to quantize the model and compare the performance to understand the benefit of LPOT.

## Key Implementation Details

- Use Keras on TensorFlow to build and train the CNN model.


- Define function and class for LPOT to quantize the CNN model.

  The LPOT can run on any Intel&reg; CPU to quantize the AI model.
  
  The quantized AI model has better inference performance than the FP32 model on Intel CPU.
  
  Specifically, it could be speeded up by the Second Generation Intel&reg; Xeon&reg; Scalable Processors and newer Xeon&reg;.
  
  
- Test the performance of the FP32 model and INT8 (quantization) model.


## Pre-requirement

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | The Second Generation Intel&reg; Xeon&reg; Scalable processor family or newer
| Software                          | Intel&reg; oneAPI AI Analytics Toolkit 2021.1 or newer
| What you will learn               | How to use LPOT tool to quantize the AI model based on TensorFlow and speed up the inference on Intel&reg; Xeon&reg; CPU
| Time to complete                  | 10 minutes

## Running Environment

### Running Samples In DevCloud (Optional)

### Run in Interactive Mode
This sample runs in a Jupyter notebook. See [Running the Sample](#running-the-sample).

### Request a Compute Node
In order to run on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
 This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

<!---Mark each compatible Node in BOLD-->
| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
| __CPU__           | __qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh__      |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |

### Running in Local Server

Please make sure the local server is installed with Ubuntu 18.04 and the following software as below guide.

For hardware, it's recommended to choose the Second Generation Intel&reg; Xeon&reg; Scalable Processors and newer Xeon&reg; processors. It will speed up the quantized model significantly. 

## Prepare Software Environment


### oneAPI

For the devcloud user, it is already installed. Please skip it.

Please install Intel&reg; AI Analytics Toolkit by referring to [Intel&reg; AI Analytics Toolkit Powered by oneAPI](
https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html). 


Intel&reg; Optimization of TensorFlow* are included in Intel&reg; AI Analytics Toolkit. So, no need to install them separately.

This sample depends on **TensorFlow* 2.2** or newer.

### Activate Intel&reg; AI Analytics Toolkit

Please change the oneAPI installed path in the following cmd, according to your installation.

In this case, we use "/opt/intel/oneapi" as exapmle.

- Activate oneAPI

```
   source /opt/intel/oneapi/setvars.sh
```

- Activate Conda Env. of Intel&reg; Optimization of TensorFlow*

  1. Show Conda Env.
  
```
conda info -e
# conda environments:
#                        
base                  *  /opt/intel/oneapi/intelpython/latest
pytorch                  /opt/intel/oneapi/intelpython/latest/envs/pytorch
pytorch-1.7.0            /opt/intel/oneapi/intelpython/latest/envs/pytorch-1.7.0
tensorflow               /opt/intel/oneapi/intelpython/latest/envs/tensorflow
tensorflow-2.3.0         /opt/intel/oneapi/intelpython/latest/envs/tensorflow-2.3.0
                         /opt/intel/oneapi/pytorch/1.7.0
                         /opt/intel/oneapi/tensorflow/2.3.0
```

  2. Activate TensorFlow Env.
  
```
conda activate tensorflow
(tensorflow) xxx@yyy:
            
```

### Install LPOT by Local Channel

```
conda install -c ${ONEAPI_ROOT}/conda_channel numpy pyyaml scikit-learn schema lpot -y
```

### Install Jupyter Notebook

```
python -m pip install notebook
```

### Install Matplotlib

```
python -m pip install matplotlib
```

## Running the Sample <a name="running-the-sample"></a>

### Startup Jupyter Notebook

The sample is drafted based on Jupyter Notebook, please run it firstly.

Steps:

```
source /opt/intel/oneapi/setvars.sh
conda activate /opt/intel/oneapi/intelpython/latest/envs/tensorflow
./run_jupyter.sh 

(tensorflow) xxx@yyy:$ [I 09:48:12.622 NotebookApp] Serving notebooks from local directory: 
...
[I 09:48:12.622 NotebookApp] Jupyter Notebook 6.1.4 is running at:
[I 09:48:12.622 NotebookApp] http://yyy:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca
[I 09:48:12.622 NotebookApp]  or http://127.0.0.1:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca
[I 09:48:12.622 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 09:48:12.625 NotebookApp] 
    
    To access the notebook, open this file in a browser:
        ...
    Or copy and paste one of these URLs:
        http://yyy:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca
     or http://127.0.0.1:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca
[I 09:48:26.128 NotebookApp] Kernel started: bc5b0e60-058b-4a4f-8bad-3f587fc080fd, name: python3
[IPKernelApp] ERROR | No such comm target registered: jupyter.widget.version

```

### Open Sample Code File

In a web browser, open link: **http://yyy:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca**. Click 'lpot_sample_tensorflow.ipynb' to start up the sample.

### Run

Next, all of practice of the sample is running in Jupyter Notebook.
