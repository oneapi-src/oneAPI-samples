# `Intel® Neural Compressor TensorFlow* Getting Started` Sample

This sample demonstrates using the Intel® Neural Compressor, which is part of the AI Tools with the with Intel® Extension for TensorFlow* to speed up inference by simplifying the process of converting the FP32 model to INT8/BF16.

| Property                 | Description |
|:---                      |:--          |
| Category                 | Getting Started |
| What you will learn      | How to use Intel® Neural Compressor tool to quantize the AI model based on TensorFlow* and speed up the inference on Intel® Xeon® CPUs |
| Time to complete         | 10 minutes |


## Purpose

This sample shows the process of building a convolution neural network (CNN) model to recognize handwritten numbers and demonstrates how to increase the inference performance by using Intel® Neural Compressor. Low-precision optimizations can speed up inference. Intel® Neural Compressor simplifies the process of converting the FP32 model to INT8/BF16. At the same time, Intel® Neural Compressor tunes the quantization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

You can achieve higher inference performance by converting the FP32 model to INT8 or BF16 model. Additionally, Intel® Deep Learning Boost (Intel® DL Boost) in Intel® Xeon® Scalable processors and Xeon® processors provides hardware acceleration for INT8 and BF16 models.

You will learn how to train a CNN model with Keras and TensorFlow*, use Intel® Neural Compressor to quantize the model, and compare the performance to see the benefit of Intel® Neural Compressor.

## Prerequisites

| Optimized for                     | Description |
|:---                               |:---         |
| OS                                | Ubuntu* 20.04 (or newer) <br> Windows 11, 10* |
| Hardware                          | Intel® Core™ Gen10 Processor <br> Intel® Xeon® Scalable Performance processors |
| Software                          | Intel® Neural Compressor, Intel® Extension for TensorFlow* |
> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

### Intel® Neural Compressor and Sample Code Versions

>**Note**: See the [Intel® Neural Compressor](https://github.com/intel/neural-compressor) GitHub repository for more information and recent changes.

This sample is updated regularly to match the Intel® Neural Compressor version in the latest AI Tools release. If you want to get the sample code for an earlier toolkit release, check out the corresponding git tag.

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

## Key Implementation Details

The sample demonstrates how to:

- Use Keras from TensorFlow* to build and train a CNN model.
- Define a function and class for Intel® Neural Compressor to
  quantize the CNN model.
  - The Intel® Neural Compressor can run on any Intel® CPU to quantize the AI model.
  - The quantized AI model has better inference performance than the FP32 model on Intel CPUs.
  - Specifically, the latest Intel® Xeon® Scalable  processors and  Xeon® processors provide hardware acceleration for such tasks.
- Test the performance of the FP32 model and INT8 (quantization) model.

## Environment Setup

You will need to download and install the following toolkits, tools, and components to use the sample.

If you have already set up the PIP or Conda environment and installed AI Tools go directly to Run the Notebook.

### 1. Get AI Tools

Required AI Tools: Intel® Neural Compressor, Intel® Extension for TensorFlow* (CPU)

If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

>**Note**: If Docker option is chosen in AI Tools Selector, refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

### 2. (Offline Installer) Activate the AI Tools bundle base environment

If the default path is used during the installation of AI Tools:

```
source $HOME/intel/oneapi/intelpython/bin/activate
```

If a non-default path is used:

```
source <custom_path>/bin/activate
```

### 3. (Offline Installer) Activate relevant Conda environment

```
conda activate tensorflow
```

### 4. Clone the GitHub repository

```
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/INC-Sample-for-Tensorflow
```

### 5. Install dependencies

Install for Jupyter Notebook:

```
pip install -r requirements.txt
```

For Jupyter Notebook, refer to [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.


## Run the Sample

> **Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/INC-Sample-for-TensorFlow#environment-setup) is completed.
>
Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:

* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip)
* [Docker](#docker)


### AI Tools Offline Installer (Validated)

#### 1. Register Conda kernel to Jupyter Notebook kernel

> **Note**: If you have done this step before, skip it.

If the default path is used during the installation of AI Tools:

```
$HOME/intel/oneapi/intelpython/envs/<offline-conda-env-name>/bin/python -m ipykernel install --user --name=tensorflow
```

If a non-default path is used:

```
<custom_path>/bin/python -m ipykernel install --user --name=tensorflow
```

#### 2. Launch Jupyter Notebook

- Option A: Launch Jupyter Notebook.

   ```
   jupyter notebook --ip=0.0.0.0
   ```

- Option B: You can launch Jupyter Notebook by running the script located in the sample code directory.

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

#### 3. Follow the instructions to open the URL with the token in your browser

#### 4. Select the Notebook

```
inc_sample_tensorflow.ipynb
```

#### 5. Change the kernel to `tensorflow`

#### 6. Run every cell in the Notebook in sequence

### Conda/PIP

> **Note**: Before running the instructions below, make sure your Conda/Python environment with AI Tools installed is activated

#### 1. Register Conda/Python kernel to Jupyter Notebook kernel

> **Note**: If you have done this step before, skip it.

For Conda:
```
<CONDA_PATH_TO_ENV>/bin/python -m ipykernel install --user --name=tensorflow
```

To know <CONDA_PATH_TO_ENV>, run conda env list and find your Conda environment path.

For PIP:

```
python -m ipykernel install --user --name=tensorflow
```

#### 2. Launch Jupyter Notebook


- Option A: Launch Jupyter Notebook.

   ```
   jupyter notebook --ip=0.0.0.0
   ```

- Option B: You can launch Jupyter Notebook by running the script located in the sample code directory.

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

#### 3. Follow the instructions to open the URL with the token in your browser

#### 4. Select the Notebook

```
inc_sample_tensorflow.ipynb
```

#### 5. Change the kernel to `tensorflow`

#### 6. Run every cell in the Notebook in sequence

### Docker

AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

## Example Output

You should see log print and images showing the performance comparison with absolute and relative data and analysis between FP32 and INT8.

Following is an example. Your data should be different with them.

```
#absolute data
throughputs_times [1, 2.51508607887295]
latencys_times [1, 0.38379207710795576]
accuracys_times [0, -0.009999999999990905]

#relative data
throughputs_times [1, 2.51508607887295]
latencys_times [1, 0.38379207710795576]
accuracys_times [0, -0.009999999999990905]
```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Related Samples

[Pytorch `Getting Started with Intel® Neural Compressor for Quantization` Sample](../INC-Quantization-Sample-for-PyTorch)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
