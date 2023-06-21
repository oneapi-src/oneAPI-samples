# `Intel® Neural Compressor TensorFlow* Getting Started*` Sample

The  `Intel® Neural Compressor TensorFlow* Getting Started*` Sample demonstrates using the Intel® Neural Compressor, which is part of the Intel® AI Analytics
Kit (AI Kit) with the with Intel® Optimizations for TensorFlow* to speed up inference by simplifying the process of converting the FP32 model to INT8/BF16.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to use Intel® Neural Compressor tool to quantize the AI model based on TensorFlow* and speed up the inference on Intel® Xeon® CPUs
| Time to complete         | 10 minutes
| Category                 | Getting Started

## Purpose

This sample shows the process of building a convolutional neural network (CNN) model to recognize handwritten numbers and demonstrates how to increase the inference performance by using Intel® Neural Compressor. Low-precision optimizations can speed up inference. Intel® Neural Compressor simplifies the process of converting the FP32 model to INT8/BF16. At the same time, Intel® Neural Compressor tunes the quantization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

You can achieve higher inference performance by converting the FP32 model to INT8 or BF16 model. Additionally, Intel® Deep Learning Boost (Intel® DL Boost) in Intel® Xeon® Scalable processors and Xeon® processors provides hardware acceleration for INT8 and BF16 models.

You will learn how to train a CNN model with Keras and TensorFlow*, use Intel® Neural Compressor to quantize the model, and compare the performance to see the benefit of Intel® Neural Compressor.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04 (or newer) <br> Windows 11, 10*
| Hardware                          | Intel® Core™ Gen10 Processor <br> Intel® Xeon® Scalable Performance processors
| Software                          | Intel® AI Analytics Toolkit (AI Kit)

### Intel® Neural Compressor and Sample Code Versions

>**Note**: See the [Intel® Neural Compressor](https://github.com/intel/neural-compressor) GitHub repository for more information and recent changes. 

This sample is updated regularly to match the Intel® Neural Compressor version in the latest Intel® AI Analytics Toolkit release. If you want to get the sample code for an earlier toolkit release, check out the corresponding git tag.

1. List the available git tags.
   ```
   git tag
   ...
   2022.3.0
   2023.0.0
   ```
2. Checkout the associated git tag.
   ```
   git checkout 2022.3.0
   ```

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

  Intel® Optimizations for TensorFlow* is included in AI Kit.

- **Jupyter Notebook**

  Install using PIP: `$pip -m install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

- **TensorFlow\* 2.2** (or newer)

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/) for information.


## Key Implementation Details

The sample demonstrates how to:

- Use Keras from TensorFlow* to build and train a CNN model.
- Define a function and class for Intel® Neural Compressor to
  quantize the CNN model.
  - The Intel® Neural Compressor can run on any Intel® CPU to quantize the AI model.
  - The quantized AI model has better inference performance than the FP32 model on Intel CPUs.
  - Specifically, the latest Intel® Xeon® Scalable  processors and  Xeon® processors provide hardware acceleration for such tasks.
- Test the performance of the FP32 model and INT8 (quantization) model.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Prepare the Environment

### On Linux*

#### Activate Conda

You can list the available conda environments using a command similar to the following

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
1. Activate the conda environment with Intel® Optimizations for TensorFlow*.

   By default, the Intel® AI Analytics Toolkit is installed in
   the `/opt/intel/oneapi` folder, which requires root privileges to manage it.

   1. If you have the root access to your oneAPI installation path:
       ```
       conda activate tensorflow
       (tensorflow) xxx@yyy:
       ```

   2. If you do not have the root access to your oneAPI installation path, clone the `tensorflow` conda environment using the following command:
      ```
      conda create --name usr_tensorflow --clone tensorflow
       ```

   3. Activate your conda environment with the following command:
      ```
      source activate usr_tensorflow
      ```
2. Install Intel® Neural Compressor from the local channel.
   ```
   conda install -c ${ONEAPI_ROOT}/conda_channel neural-compressor -y --offline
   ```

#### Configure Jupyter Notebook

1. Create a new kernel for the Jupyter notebook based on your activated conda environment.
   ```
   conda install ipykernel
   python -m ipykernel install --user --name usr_tensorflow
   ```
   This step is optional if you plan to open the notebook on your local server.

### On Windows*

#### Configure Conda

1. Configure Conda for **user_tensorflow** by entering commands similar to the following:
   ```
   conda deactivate
   conda env remove -n user_tensorflow
   conda create -n user_tensorflow python=3.9 -y
   conda activate user_tensorflow
   conda install -n user_tensorflow pycocotools -c esri -y
   conda install -n user_tensorflow neural-compressor tensorflow -c conda-forge -c intel -y
   conda install -n user_tensorflow jupyter runipy notebook -y
   ```

## Run the `Intel® Neural Compressor TensorFlow* Getting Started*` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html)*.

### On Linux

1. Ensure you activate the conda environment.
   ```
   source /opt/intel/oneapi/setvars.sh
   conda activate tensorflow
   ```
   or
   ```
   conda activate usr_tensorflow
   ```
2. Change to the sample directory. 
3. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0
   ```
4. Alternatively, you can launch Jupyter Notebook by running the script located in the sample code directory.
	```
	./run_jupyter.sh
	```
	The Jupyter Server shows the URLs of the web application in your terminal.

	```
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
   In a web browser, open the link that the Jupyter server displayed when you started it. For example:
   **http://yyy:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca**.

5. Locate and select the Notebook.
   ```
   inc_sample_tensorflow.ipynb
   ```
6. Change the kernel to **user_tensorflow**.
7. Run every cell in the Notebook in sequence.


#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

### On Intel® DevCloud

1. Open Jupyter Hub in a browser: [oneAPI JupyterHub](https://jupyter.oneapi.devcloud.intel.com/).

2. Navigate to the `inc_sample_tensorflow.ipynb` file and open it.

3. Change the Kernel to **user_tensorflow**.

4. Run every cell in the Notebook in sequence.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).