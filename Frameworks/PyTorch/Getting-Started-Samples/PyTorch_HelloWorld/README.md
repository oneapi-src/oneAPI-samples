# PyTorch HelloWorld
The official PyTorch has been optimized using Intel(R) Deep Neural Networks Library (Intel(R) DNNL) primitives by default. This sample shows how to train a PyTorch model and run the inference with Intel DNNL enabled. 


## Key implementation details
This Hello World sample code is implemented for CPU using the Python language. 

*Please* **export the environment variable `MKLDNN_VERBOSE=1`** *to display the deep learning primitives trace during execution.*

### Notes
 - The test dataset is inherited from `torch.utils.data.Dataset`.
 - The model is inherited from `torch.nn.Module`.
 - For the inference portion, `to_mkldnn()` function in `torch.utils.mkldnn` can accelerate performance by eliminating data reorders between operations which are supported by Intel(R) DNNL.


## Pre-requirement

PyTorch is ready for use once you finish the Intel AI Analytics Toolkit installation, and have run post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation, and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

## Activate conda environment With Root Access

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the setvars.sh script. Then navigate in linux shell to your oneapi installation path, typically `~/intel/inteloneapi`. Activate the conda environment with the following command:

#### Linux
```
source activate pytorch
```

## Activate conda environment Without Root Access (Optional)

By default, the Intel AI Analytics toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### Linux
```
conda create --name user_pytorch --clone pytorch
```

Then activate your conda environment with the following command:

```
source activate user_pytorch
```


## How to Build and Run 

To run the program on Linux*, Windows* and MacOS*, simply type the following command in the terminal with Python installed:

```
    python PyTorch_Hello_World.py
```

With successful execution, it will print out `PASSED_CICD` in the terminal.

You will see the DNNL verbose trace after exporting the `MKLDNN_VERBOSE`:

```
    export MKLDNN_VERBOSE=1
```

Please go to https://intel.github.io/mkl-dnn/dev_guide_verbose.html to find more information about the mkldnn log. 

## License  
This code sample is licensed under MIT license.