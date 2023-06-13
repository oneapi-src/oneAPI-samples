# `Fine-tuning Text Classification Model with Intel® Neural Compressor (INC)` Sample

The `Fine-tuning Text Classification Model with Intel® Neural Compressor (INC)` sample demonstrates how to fine-tune BERT tiny model for emotion classification task using Quantization Aware Training (QAT) from Intel® Neural Compressor.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to fine-tune text model using Intel® Neural Compressor Quantization Aware Training
| Time to complete      | 10 minutes
| Category              | Concepts and Functionality

Intel® Neural Compressor simplifies the process of converting the FP32 model to INT8/BF16. At the same time, Intel® Neural Compressor tunes the quantization method to reduce the accuracy loss, which is a big blocker for low-precision inference as part of Intel® AI Analytics Toolkit (AI Kit).

## Purpose

This sample shows how to fine-tune text model for emotion classification on pre-trained `bert-tiny` model from Hugging Face and how to perform fine-tuning using Intel® Neural Compressor Quantization Aware Training. Fine-tuning allows you to speed up operations on processors with INT8 data format and specialized computer instructions. The INT8 data format uses quarter the bit width of floating-point-32 (FP32), lowering the amount of memory needed and execution time to process with minimum to zero accuracy loss.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04 (or newer)
| Hardware                | Intel® Xeon® Scalable Processor family
| Software                | Intel® Neural Compressor (INC)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

- **Additional Packages**

  You will need to install the additional packages in requirements.txt.

  ```
  pip install -r requirements.txt
  ```

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See *[Intel® DevCloud for oneAPI](https://DevCloud.intel.com/oneapi/get_started/)* for information.

## Key Implementation Details

This code sample implements fine-tuning process for text classification using Intel® Neural Compressor quantization aware training.

The sample tutorial contains one Jupyter Notebook and a Python script. You can use either.

### Jupyter Notebook

| Notebook                                                 | Description
|:---                                                      |:---
|`mINC_QuantizationAwareTraining_TextClassification.ipynb` | Performs chat creation with Intel® Extension for PyTorch* quantization and provides interface for interactions in Jupyter Notebook.

### Python Scripts

| Script                                                   | Description
|:---                                                      |:---
|`INC_QuantizationAwareTraining_TextClassification.py`     | The script performs chat creation with Intel® Extension for PyTorch* quantization and provides simple interactions based on prepared input.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Fine-tuning Text Classification Model with Intel® Neural Compressor` Sample

### On Linux*

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

#### Activate Conda

1. Activate the Conda environment.
   ```
   conda activate pytorch
   ```
2. Activate Conda environment without Root access (Optional).

   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

   ```
   conda create --name user_pytorch --clone pytorch
   conda activate user_pytorch
   ```

#### Running the Jupyter Notebook

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
   ```
   INC_QuantizationAwareTraining_TextClassification.ipynb
   ```
5. Change your Jupyter Notebook kernel to corresponding environment.
6. Run every cell in the Notebook in sequence.

#### Running on the Command Line (Optional)

1. Change to the sample directory.
2. Run the script.
   ```
   python INC_QuantizationAwareTraining_TextClassification.py
   ```

### Run the Sample on Intel® DevCloud (Optional)

1. If you do not already have an account, request an Intel® DevCloud account at [*Create an Intel® DevCloud Account*](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).
2. On a Linux* system, open a terminal.
3. SSH into Intel® DevCloud.
   ```
   ssh DevCloud
   ```
   > **Note**: You can find information about configuring your Linux system and connecting to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://DevCloud.intel.com/oneapi/get_started).

4. Follow the instructions to open the URL with the token in your browser.
5. Locate and select the Notebook.
   ```
   INC_QuantizationAwareTraining_TextClassification.ipynb
   ````
6. Change your Jupyter Notebook kernel to corresponding environment.
7. Run every cell in the Notebook in sequence.

### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample shows statistics for training, quantization information and classification results, before and after fine-tuning.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).