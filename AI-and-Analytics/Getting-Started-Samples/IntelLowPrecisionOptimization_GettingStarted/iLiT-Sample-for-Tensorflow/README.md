# Intel(R) Low Precision Optimization Tool (iLiT) Sample for Tensorflow

## Background
Low-precision inference can speed up infernece obviously, by converting fp32 model to int8 or bf16 model. Intel provides Intel(R) Deep Learning Boost techonoloy in the Second Generation Intel(R) Xeon(R) Scalable Processors and newer Xeon(R), which supports to speed up int8 and bf16 model by hardware.

Intel(R) Low Precision Optimization Tool (iLiT) helps user to simple the processing to convert fp32 model to int8/bf16.

In same time, iLiT will tune the quanization method to reduce the accuracy loss, which is big blocker for low-precision inference.

iLiT is released in Intel(R) AI Analytics Toolkit and works with Intel(R) Optimizition of Tensorflow*.

Please refer to official website for detailed info and news: [https://github.com/intel/lp-opt-tool](https://github.com/intel/lp-opt-tool)

## License

This code sample is licensed under MIT license.

## Purpose
In this sample, we will show a whole process to build up a CNN model to recognize hand writing number and speed up it by iLiT.

We will learn how to train a CNN model based on Keras with Tensorflow, use iLiT to quantize the model and compare the performance, to learn the benefit of iLiT.

## Key Implementation Details

- Use Keras on Tensorflow to build and train CNN model.


- Define function and class for iLiT to quantize the CNN model.

  The iLiT can run on any Intel(R) CPU to quantize the AI model.
  
  The quantized AI model has better inference performance than FP32 model on Intel CPU.
  
  Speically, it could be speeded up by the Second Generation Intel(R) Xeon(R) Scalable Processors and newer Xeon(R).
  
  
- Test the performance of FP32 model and INT8 (quantization) model.


## Pre-requirement

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | The Second Generation Intel(R) Xeon(R) Scalable processor family or newer
| Software                          | Intel(R) oneAPI AI Analytics Toolkit
| What you will learn               | How to use iLiT tool to quantize the AI model based on Tensorflow and speed up the inference on Intel(R) Xeon(R) CPU
| Time to complete                  | 10 minutes

## Running Environment

### Running in Devcloud

If running a sample in the Intel DevCloud, please follow the below steps to build the python environment. Also remember that you must specify the compute node (CPU) as well whether to run in batch or interactive mode. For more information see the [Intel(R) oneAPI AI Analytics Toolkit Get Started Guide] https://devcloud.intel.com/oneapi/get-started/analytics-toolkit/)

### Running in Local Server

Please make sure the local server is installed with Ubuntu 18.04 and following software as below guide.

For hardware, it's recommended to choose the Second Generation Intel(R) Xeon(R) Scalable Processors and newer Xeon(R). It will speed up the quantized model specially. 

## Prepare Software Environment


### oneAPI

For devcloud user, it is already installed. Please skip it.

Please install Intel(R) AI Analytics Toolkit by refer to [Intel(R) AI Analytics Toolkit(Beta) Powered by oneAPI](
https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html). 


Intel(R) Optimizition of Tensorflow* are included in Intel(R) AI Analytics Toolkit. So, no need to install them seperately.

This sample depends on **Tensorflow* 2.2**.

### Activate Intel(R) AI Analytics Toolkit

Please change the oneAPI installed path in following cmd, according to your installation.

In this case, we use "/opt/intel/oneapi" as exapmle.

- Activate oneAPI

```
   source /opt/intel/oneapi/setvars.sh
```

- Activate Conda Env. of Intel(R) Optimizition of Tensorflow*

  1. Show Conda Env.
  
```
conda info -e
# conda environments:
#                        
base                  *  /opt/intel/oneapi/intelpython/latest
pytorch                  /opt/intel/oneapi/intelpython/latest/envs/pytorch
pytorch-1.5.0            /opt/intel/oneapi/intelpython/latest/envs/pytorch-1.5.0
tensorflow               /opt/intel/oneapi/intelpython/latest/envs/tensorflow
tensorflow-2.2.0         /opt/intel/oneapi/intelpython/latest/envs/tensorflow-2.2.0
                         /opt/intel/oneapi/pytorch/1.5.0
                         /opt/intel/oneapi/tensorflow/2.2.0
```

  2. Activate Tensorflow Env.
  
```
conda activate tensorflow
(tensorflow) xxx@yyy:
            
```

### Install iLiT by Local Channel

```
cd /opt/intel/oneapi/iLiT/latest
sudo ./install_iLiT.sh
```

### Install Jupyter Notebook

```
python -m pip install notebook
```

### Install Matplotlib

```
python -m pip install matplotlib
```

## Running the Sample

### Start up Jupyter Notebook

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

In a web browser, open link: **http://yyy:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca**. Click 'ilit_sample_tensorflow.ipynb' to start up the sample.

### Run

Next, all of pratice of the sample is running in Jupyter Notebook.
