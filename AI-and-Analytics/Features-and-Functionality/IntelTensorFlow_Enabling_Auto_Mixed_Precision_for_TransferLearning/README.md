# `Enable Auto-Mixed Precision for Transfer Learning with TensorFlow*` Sample

The `Enable Auto-Mixed Precision for Transfer Learning with TensorFlow*` sample guides you through the process of enabling auto-mixed precision to use low-precision datatypes, like bfloat16, for transfer learning with TensorFlow* (TF).

The sample demonstrates the end-to-end pipeline tasks typically performed in a deep learning use-case: training (and retraining), inference optimization, and serving the model with TensorFlow Serving.

| Area                    | Description
|:---                     |:---
| What you will learn     | Enable Auto-Mixed Precision for Transfer Learning with TensorFlow*
| Time to complete        | 30 minutes
| Category                | Code Optimization

## Purpose

Through the implementation of end-to-end deep learning example, this sample demonstrates important concepts:
- The benefits of using auto-mixed precision to accelerate tasks like transfer learning, with minimal changes to existing scripts.
- The importance of inference optimization on performance.
- The ease of using Intel® optimizations in TensorFlow, which are enabled by default in 2.9.0 and newer.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 or newer
| Hardware                          | Intel® Xeon® Scalable processor family or newer
| Software                          | Intel® AI Analytics Toolkit (AI Kit)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

- **TensorFlow Serving**

  See *TensorFlow Serving* [*Installation*](https://www.tensorflow.org/tfx/serving/setup) for detailed installation options.

- **Other dependencies**

  Install using PIP and the `requirements.txt` file supplied with the sample: `$pip install -r requirements.txt`. <br> The `requirements.txt` file contains the necessary dependencies to run the Notebook.

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/) for information.

## Key Implementation Details

The sample tutorial contains one Jupyter Notebook and two Python scripts.

### Jupyter Notebook

| Notebook                                                         | Description
|:---                                                              |:---
|`enabling_automixed_precision_transfer_learning_tensorflow.ipynb` | Enabling Auto-Mixed Precision for Transfer Learning with TensorFlow

### Python Scripts

| Script                   | Description
|:---                      |:---
|`freeze_optimize_v2.py`   |The script optimizes a pre-trained TensorFlow model PB file.
|`tf_benchmark.py`         |The script measures inference performance of a model using dummy data.

## Run the Enable Auto-Mixed Precision for Transfer Learning with TensorFlow* 

### On Linux*

1. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0
   ```
2. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the Notebook.
   ```
   enabling_automixed_precision_transfer_learning_tensorflow.ipynb
   ````
4. Change your Jupyter Notebook kernel to **tensorflow** or **intel-tensorflow**.
5. Run every cell in the Notebook in sequence.


### Run the Sample on Intel® DevCloud (Optional)

1. If you do not already have an account, request an Intel® DevCloud account at [*Create an Intel® DevCloud Account*](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).
2. On a Linux* system, open a terminal.
3. SSH into Intel® DevCloud.
   ```
   ssh DevCloud
   ```
   > **Note**: You can find information about configuring your Linux system and connecting to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started).
4. Follow the instructions to open the URL with the token in your browser.
5. Locate and select the Notebook.
   ```
   enabling_automixed_precision_transfer_learning_tensorflow.ipynb
   ````
6. Change the kernel to **tensorflow** or **intel-tensorflow**.
7. Run every cell in the Notebook in sequence.


#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Example Output

You will see diagrams comparing performance and analysis. This includes performance comparison for training speedup obtained by enabling auto-mixed precision and inference speedup obtained by optimizing the saved model for inference.

For performance analysis, you will see histograms showing different Tensorflow* operations in the analyzed pre-trained model pb file.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).