# `Intel® Model Zoo` Sample
This code example provides a sample code to run ResNet50 inference on Intel's pretrained FP32 and Int8 model

## Purpose
  - Demonstrate the AI workloads and deep learning models Intel has optimized and validated to run on Intel hardware
  - Show how to efficiently execute, train, and deploy Intel-optimized models
  - Make it easy to get started running Intel-optimized models on Intel hardware in the cloud or on bare metal

***DISCLAIMER: These scripts are not intended for benchmarking Intel platforms.
For any performance and/or benchmarking information on specific Intel platforms, visit [https://www.intel.ai/blog](https://www.intel.ai/blog).***
## Key implementation details
The example uses Intel's pretrained model published as part of [Intel Model Zoo](https://github.com/IntelAI/models). The example also illustrates how to utilize TensorFlow and MKL run time settings to maximize CPU performance on ResNet50 workload.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Running Samples on the Intel&reg; DevCloud
If you are running this sample on the DevCloud, skip the Pre-requirements and go to the [Activate Conda Environment](#activate-conda) section.

## Pre-requirements (Local or Remote Host Installation)

TensorFlow* is ready for use once you finish the Intel® AI Analytics Toolkit installation and have run the post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Intel&reg; oneAPI AI Analytics Toolkit Get Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

## Activate conda environment With Root Access<a name="activate-conda"></a>

Navigate the Linux shell to your oneapi installation path, typically `/opt/intel/oneapi`. Activate the conda environment with the following command:

#### Linux
```
conda activate tensorflow
```


## Activate conda environment Without Root Access (Optional)

By default, the Intel® oneAPI AI Analytics Toolkit (AI Kit) is installed in the `/opt/intel/oneapi` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

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
You can view the available Model Zoo release versions for the Intel AI Analytics Toolkit:
```
ls /opt/intel/oneapi/modelzoo
1.8.0  latest
```
Then browse to the preferred [Intel Model Zoo](https://github.com/IntelAI/models/tree/master/benchmarks) release version location to run inference for ResNet50 or another supported topology.
```
cd /opt/intel/oneapi/modelzoo/latest
```

## Install Jupyter Notebook*
```
conda install jupyter nb_conda_kernels
```

## How to Build and Run
1. Go to the code example location.<br>
2. If you have GUI support, enter the command `jupyter notebook`. <br>
or<br>
a. If you do not have GUI support, open a remote shell and enter command `jupyter notebook --no-browser --port=8888`.<br>
b. Open the command prompt where you have GUI support, and forward the port from host to client.<br>
c. Enter `ssh -N -f -L localhost:8888:localhost:8888 <userid@hostname>`<br>
d. Copy-paste the URL address from the host into your local browser to open the jupyter console.<br>
3. Go to `ResNet50_Inference.ipynb` and run each cell to create synthetic data and run int8 inference.

---
**NOTE**

In the jupyter page, be sure to select the correct kernel. In this example, select 'Kernel' -> 'Change kernel' -> Python [conda env:tensorflow].

---

### **Request a Compute Node**
In order to run on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
 This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

<!---Mark each compatible Node in BOLD-->
| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
| CPU               | qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh          |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |
