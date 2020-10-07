# Intel Model Zoo Sample
This code example provides a sample code to run ResNet50 inference on Intel's pretrained FP32 and Int8 model

## Purpose
  - Demonstrate the AI workloads and deep learning models Intel has optimized and validated to run on Intel hardware
  - Show how to efficiently execute, train, and deploy Intel-optimized models
  - Make it easy to get started running Intel-optimized models on Intel hardware in the cloud or on bare metal

***DISCLAIMER: These scripts are not intended for benchmarking Intel platforms. 
For any performance and/or benchmarking information on specific Intel platforms, visit [https://www.intel.ai/blog](https://www.intel.ai/blog).***
## Key implementation details
The example uses Intel's pretrained model published as part of [Intel Model Zoo](https://github.com/IntelAI/models). The example also illustrates how to utilize TensorFlow and MKL run time settings to maximize CPU performance on ResNet50 workload

## License  
This code sample is licensed under MIT license

## Pre-requirement

TensorFlow is ready for use once you finish the Intel AI Analytics Toolkit installation, and have run post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation, and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

## Activate conda environment With Root Access

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the setvars.sh script. Then navigate in linux shell to your oneapi installation path, typically `/opt/intel/oneapi`. Activate the conda environment with the following command:

#### Linux
```
conda activate tensorflow
```


## Activate conda environment Without Root Access (Optional)

By default, the Intel AI Analytics toolkit is installed in the `/opt/intel/oneapi` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### Linux
```
conda create --name user_tensorflow --clone tensorflow
```

Then activate your conda environment with the following command:

```
conda activate user_tensorflow
```

## Navigate to Intel Model Zoo

Navigate to the Intel Model Zoo source directory. It's located in your oneapi installation path, typically `/opt/intel/oneapi/modelzoo`.
You can view the available Model Zoo release versions for the Intel AI Analytics toolkit:
```
ls /opt/intel/oneapi/modelzoo
1.8.0  latest
```
Then browse to the preferred [Intel Model Zoo](https://github.com/IntelAI/models/tree/master/benchmarks) release version location to run inference for ResNet50 or another supported topology.
```
cd /opt/intel/oneapi/modelzoo/latest
```

## Install Jupyter Notebook 
```
conda install jupyter nb_conda_kernels
```

## How to Build and Run 
1. Go to the code example location<br>
2. Enter command `jupyter notebook` if you have GUI support <br>
or<br>
2a. Enter command `jupyter notebook --no-browser --port=8888` on a remote shell <br>
2b. Open the command prompt where you have GUI support, and forward the port from host to client<br>
2c. Enter `ssh -N -f -L localhost:8888:localhost:8888 <userid@hostname>`<br>
2d. Copy-paste the URL address from the host into your local browser to open the jupyter console<br>
3. Go to `ResNet50_Inference.ipynb` and run each cell to create sythetic data and run int8 inference

Note, In jupyter page, please choose the right 'kernel'. In this case, please choose it in menu 'Kernel' -> 'Change kernel' -> Python [conda env:tensorflow]
