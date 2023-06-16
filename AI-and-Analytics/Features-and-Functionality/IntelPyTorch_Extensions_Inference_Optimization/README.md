# `Optimize PyTorch* Models using Intel® Extension for PyTorch* (IPEX)` Sample

This notebook guides you through the process of extending your PyTorch* code with Intel® Extension for PyTorch* (IPEX) with optimizations to achieve performance boosts on Intel® hardware.

| Area                  | Description
|:---                   |:---
| What you will learn   | Applying IPEX Optimizations to a PyTorch workload in a step-by-step manner to gain performance boost
| Time to complete      | 30 minutes
| Category              | Code Optimization

## Purpose

This sample notebook shows how to get started with Intel® Extension for PyTorch (IPEX) for sample Computer Vision and NLP workloads.

The sample starts by loading two models from the PyTorch hub: **Faster-RCNN** (Faster R-CNN) and **distilbert** (DistilBERT). After loading the models, the sample applies sequential optimizations from IPEX and examines performance gains for each incremental change.

You can make code changes quickly on top of existing PyTorch code to obtain the performance speedups for model inference.

## Prerequisites

| Optimized for          | Description
|:---                    |:---
| OS                     | Ubuntu* 18.04 or newer
| Hardware               | Intel® Xeon® Scalable processor family
| Software               | Intel® AI Analytics Toolkit (AI Kit)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts. This sample assumes you have **Matplotlib** installed.


- **Jupyter Notebook**

  Install using PIP: `pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

- **Transformers - Hugging Face**

  Install using PIP: `pip install transformers`

### For Intel® DevCloud

Most of necessary tools and components are already installed in the environment. You do not need to install additional components. See [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/) for information. You would need to install the Hugging Face Transformers library using pip as shown above.

## Key Implementation Details

This sample tutorial contains one Jupyter Notebook and one Python script.

### Jupyter Notebook

| Notebook                                 | Description
|:---                                      |:---
|`optimize_pytorch_models_with_ipex.ipynb` |Gain performance boost during inference using IPEX.

### Python Script

| Script              | Description
|:---                 |:---
|`resnet50.py`        |The script optimizes a Faster R-CNN model to be used with IPEX Launch Script.


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Optimize PyTorch* Models using Intel® Extension for PyTorch* (IPEX)` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

#### Activate Conda

1. Activate the Conda environment.

    ```
    conda activate pytorch
    ```

   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

#### Activate Conda without Root Access (Optional)

You can choose to activate Conda environment without root access.

1. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using commands similar to the following.

   ```
   conda create --name user_pytorch --clone pytorch
   conda activate user_pytorch
   ```

#### Run the NoteBook

1. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0
   ```
2. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the Notebook.
   ```
   optimize_pytorch_models_with_ipex.ipynb
   ```
4. Change the kernel to **PyTorch (AI Kit)**.
5. Run every cell in the Notebook in sequence.

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

### Run the Sample on Intel® DevCloud (Optional)

1. If you do not already have an account, request an Intel® DevCloud account at [*Create an Intel® DevCloud Account*](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).
2. On a Linux* system, open a terminal.
3. SSH into Intel® DevCloud.
   ```
   ssh DevCloud
   ```
   > **Note**: You can find information about configuring your Linux system and connecting to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started).
4. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the Notebook.
   ```
   optimize_pytorch_models_with_ipex.ipynb
   ```
4. Change the kernel to **PyTorch (AI Kit)**.
7. Run every cell in the Notebook in sequence.

## Example Output

Users should be able to see some diagrams for performance comparison and analysis for inference speedup obtained by enabling IPEX optimizations.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).