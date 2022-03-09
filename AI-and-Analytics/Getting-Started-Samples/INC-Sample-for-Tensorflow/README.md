# Intel® Neural Compressor Sample for TensorFlow*

Low-precision optimizations can speed up inference. You can achieve
higher inference performance by converting the FP32 model to INT8 or
BF16 model. Additionally, Intel&reg; Deep Learning Boost technology in
the Second Generation Intel&reg; Xeon&reg; Scalable processors and
newer Xeon&reg; processors provides hardware acceleration for INT8 and
BF16 models.

Intel&reg; Neural Compressor simplifies the process of converting the
FP32 model to INT8/BF16.

At the same time, Intel&reg; Neural Compressor tunes the quanization
method to reduce the accuracy loss, which is a big blocker for
low-precision inference.

Intel&reg; Neural Compressor is part of Intel&reg; oneAPI AI Analytics
Kit and works with Intel&reg; Optimizations for TensorFlow*.

Refer to the official web site for detailed information and news:
[https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)


## Purpose

This sample shows the whole process of building a convolutional neural
network (CNN) model to recognize handwritten numbers and increasing
the inference performance by using Intel&reg; Neural Compressor.

We will learn how to train a CNN model with Keras and TensorFlow,
use Intel&reg; Neural Compressor to quantize the model, and compare the
performance to see the benefit of Intel&reg; Neural Compressor.


## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 or later
| Hardware                          | The Second Generation Intel&reg; Xeon&reg; Scalable processor family or newer Xeon&reg; processors
| Software                          | Intel&reg; oneAPI AI Analytics Toolkit 2021.1 or later
| What you will learn               | How to use Intel&reg; Neural Compressor tool to quantize the AI model based on TensorFlow* and speed up the inference on Intel&reg; Xeon&reg; CPUs
| Time to complete                  | 10 minutes


### Intel® Neural Compressor and Sample Code Versions

This sample code is always updated for the Intel® Neural Compressor
version in the latest Intel® oneAPI AI Analytics Kit release.

If you want to get the sample code for an earlier toolkit release,
checkout the corresponding git tag.

List the available git tags:

```bash
git tag

2021.1-beta08
2021.1-beta09
2021.1-beta10
```

Checkout a git tag:

```bash
git checkout 2021.1-beta10
```


## Key Implementation Details

- Use Keras from TensorFlow* to build and train a CNN model.


- Define a function and class for Intel&reg; Neural Compressor to
  quantize the CNN model.

  The Intel&reg; Neural Compressor can run on any Intel&reg; CPU to
  quantize the AI model.

  The quantized AI model has better inference performance than the
  FP32 model on Intel CPUs.

  Specifically, the Second Generation Intel&reg; Xeon&reg; Scalable
  processors and newer Xeon&reg; processors provide hardware
  acceleration for such tasks.


- Test the performance of the FP32 model and INT8 (quantization) model.


## Prepare Software Environment

You can run this sample in a Jupyter notebook on your local computer
or in the Intel&reg; DevCloud.

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

1. Install Intel® oneAPI AI Analytics Toolkit.

   If you use the Intel&reg; DevCloud, skip this step. The toolkit is
   already installed for you.

   For installation instructions, refer to [Intel&reg; AI Analytics Toolkit Powered by oneAPI](
https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html).

   Intel&reg; Optimizations for TensorFlow* is included in Intel&reg;
   AI Analytics Toolkit. So, you do not have to install it separately.

   This sample depends on **TensorFlow* 2.2** or newer.

2. Activate the conda environment with Intel&reg; Optimizations for TensorFlow*.

   You can list the available conda environments with the following command:

   ```bash
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

   By default, the Intel® oneAPI AI Analytics Toolkit is installed in
   the `/opt/intel/oneapi` folder, which requires root privileges to manage it.

   - If you have the root access to your oneAPI installation path:

     ```
     conda activate tensorflow
     (tensorflow) xxx@yyy:
     ```

   - If you do not have the root access to your oneAPI installation
     path, clone the `tensorflow` conda environment using the following
     command:

     ```bash
     conda create --name usr_tensorflow --clone tensorflow
     ```

     Then activate your conda environment with the following command:

     ```bash
     source activate usr_tensorflow
     ```

3. Install Intel&reg; Neural Compressor from the local channel.

   ```bash
   conda install -c ${ONEAPI_ROOT}/conda_channel neural-compressor -y --offline
   ```

4. Install Jupyter Notebook.

   Skip this step if you are working in the DevCloud.

   ```bash
   python -m pip install notebook
   ```

5. Create a new kernel for the Jupyter notebook based on your activated conda environment.

   ```bash
   conda install ipykernel
   python -m ipykernel install --user --name usr_tensorflow
   ```

   This step is optional if you plan to open the notebook on your local server.


## Run the Sample <a name="running-the-sample"></a>

You can run the Jupyter notebook with the sample code on your local server or use Intel® DevCloud.


### Run the Sample on Local Server

To open the Jupyter notebook on your local server:

1. Make sure you activate the conda environment.

   ```bash
   source /opt/intel/oneapi/setvars.sh
   conda activate tensorflow
   ```

   or

   ```bash
   conda activate usr_tensorflow
   ```

2. Start the Jupyter notebook server.

   Run the `run_jupyter.sh` script that is located in the sample code directory:

	```bash
	./run_jupyter.sh
	```

	The jupyter server prints the URLs of the web aplication in your terminal.

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

2. In a web browser, open the link that the Jupyter server displayed when you started it. For example:
   **http://yyy:8888/?token=146761d9317552c43e0d6b8b6b9e1108053d465f6ca32fca**.

3. In the Notebook Dashboard, click `inc_sample_tensorflow.ipynb` to open the notebook.

4. Run the sample code and read the explanations in the notebook.


### Run the Sample in the Intel&reg; DevCloud

1. Open the following link in your browser:
   https://jupyter.oneapi.devcloud.intel.com/

2. In the Notebook Dashboard, navigate to the `inc_sample_tensorflow.ipynb` file and open it.

3. To change the kernel, click **Kernel** > **Change kernel** > **usr_tensorflow**.

4. Run the sample code and read the explanations in the notebook.


## Build and Run Additional Samples
Several sample programs are available for you to try, many of which
can be compiled and run in a similar fashion to this Intel&reg; Neural
Compressor sample for Tensorflow. Experiment with running the various
samples on different kinds of compute nodes or adjust their source
code to experiment with different workloads.

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
