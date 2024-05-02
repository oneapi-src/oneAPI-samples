# `Quantizing Transformer Model using Intel® Extension for Transformers (ITREX)` Sample

The `Quantizing Transformer Model using Intel® Extension for Transformers (ITREX)` sample illustrates the process of quantizing the `Intel/neural-chat-7b-v3-3` language model. This model, a fine-tuned iteration of *Mistral-7B*, undergoes quantization utilizing Weight Only Quantization (WOQ) techniques provided by Intel® Extension for Transformers (ITREX). 

By leveraging WOQ techniques, developers can optimize the model's memory footprint and computational efficiency without sacrificing performance or accuracy. This sample serves as a practical demonstration of how ITREX empowers users to maximize the potential of transformer models in various applications, especially in resource-constrained environments.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to quantize transformer models using Intel® Extension for Transformers (ITREX)
| Time to complete      | 20 minutes
| Category              | Concepts and Functionality

Intel® Extension for Transformers (ITREX) serves as a comprehensive toolkit tailored to enhance the performance of GenAI/LLM (General Artificial Intelligence/Large Language Models) workloads across diverse Intel platforms. Among its key features is the capability to seamlessly quantize transformer models to 4-bit or 8-bit integer precision.

This quantization functionality not only facilitates significant reduction in memory footprint but also offers developers the flexibility to fine-tune the quantization method. This customization empowers developers to mitigate accuracy loss, a crucial concern in low-precision inference scenarios. By striking a balance between memory efficiency and model accuracy, ITREX enables efficient deployment of transformer models in resource-constrained environments without compromising on performance or quality.

## Purpose

This sample demonstrates how to quantize a pre-trained language model, specifically the `Intel/neural-chat-7b-v3-3` model from Intel. Quantization enables more memory-efficient inference, significantly reducing the model's memory footprint. 

Using the INT8 data format, which employs a quarter of the bit width of floating-point-32 (FP32), memory usage can be lowered by up to 75%. Additionally, execution time for arithmetic operations is reduced. The INT4 data type takes memory optimization even further, consuming 8 times less memory than FP32. 

Quantization thus offers a compelling approach to deploying language models in resource-constrained environments, ensuring both efficient memory utilization and faster inference times.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 22.04.3 LTS (or newer)
| Hardware                | Intel® Xeon® Scalable Processor family
| Software                | Intel® Extension for Transformers (ITREX)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux*](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

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

This code sample showcases the implementation of quantization for memory-efficient text generation utilizing Intel® Extension for Transformers (ITREX).

The sample includes both a Jupyter Notebook and a Python Script. While the notebook serves as a tutorial for learning purposes, it's recommended to use the Python script in production setups for optimal performance.

### Jupyter Notebook

| Notebook                                      | Description
|:---                                           |:---
|`quantize_transformer_models_with_itrex.ipynb` | This notebook provides detailed steps for performing INT4/INT8 quantization of transformer models using Intel® Extension for Transformers (ITREX). It's designed to aid understanding and experimentation.|

### Python Script

| Script                                      | Description
|:---                                         |:---
|`quantize_transformer_models_with_itrex.py`  | The Python script conducts INT4/INT8 quantization of transformer models leveraging Intel® Extension for Transformers (ITREX). It allows text generation based on an initial prompt, giving users the option to select either `INT4` or `INT8` quantization. |

These components offer flexibility for both learning and practical application, empowering users to harness the benefits of quantization for transformer models efficiently.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `intelpython` environment's *activate* script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Quantizing Transformer Model using Intel® Extension for Transformers (ITREX)` Sample

### On Linux*

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing the `intelpython` environment's *activate* script in the root of your oneAPI installation.
>
> Linux*:
> - For POXIS shells, run: `source ${HOME}/intel/oneapi/intelpython/bin/activate`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source ${HOME}/intel/oneapi/intelpython/bin/activate ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

#### Activate Conda

1. Activate the Conda environment.
   ```
   conda activate pytorch
   ```
2. Activate Conda environment without Root access (Optional).

   By default, the AI Kit is installed in the `<home>/intel/oneapi/intelpython` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

   ```
   conda create --name user_pytorch --clone pytorch
   conda activate user_pytorch
   ```

#### Installing Dependencies

1. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

This script will automatically install all the required dependencies.


#### Using Jupyter Notebook

1. Navigate to the sample directory in your terminal.
2. Launch Jupyter Notebook with the following command:
   ```
   jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions provided in the terminal to open the URL with the token in your web browser.
4. Locate and select the notebook file named `quantize_transformer_models_with_itrex.ipynb`.
5. Ensure that you change the Jupyter Notebook kernel to the corresponding environment.
6. Run each cell in the notebook sequentially.

#### Running on the Command Line (for deployment)

1. Navigate to the sample directory in your terminal.
2. Execute the script using the following command:
   ```bash
   OMP_NUM_THREADS=<number of physical cores> numactl -m <node index> -C <CPU list> python quantize_transformer_models_with_itrex.py
   ```

   **Note:** You can use the command `numactl -H` to identify the number of nodes, node indices, and CPU lists. Additionally, the `lscpu` command provides information about the number of physical cores available.

   For example, if you have 8 physical cores, 1 node (index 0), and want to use CPUs 0-3, you would run:
   ```bash
   OMP_NUM_THREADS=4 numactl -m 0 -C 0-3 python quantize_transformer_models_with_itrex.py
   ```

   It is generally recommended to utilize all available physical cores and CPUs within a single node. Thus, you can simplify the command as follows:
   ```bash
   OMP_NUM_THREADS=<number of physical cores> numactl -m 0 -C all python quantize_transformer_models_with_itrex.py
   ```

Here are some examples demonstrating how to use the `quantize_transformer_models_with_itrex.py` script:

1. Quantize the model to INT4, and specify a maximum number of new tokens:
   ```bash
   OMP_NUM_THREADS=<number of physical cores> numactl -m 0 -C all \ 
   python quantize_transformer_models_with_itrex.py \
    --model_name "Intel/neural-chat-7b-v3-1" \
    --quantize "int8" \
    --max_new_tokens 100
   ```

2. Quantize the model to INT4, disable Neural Speed, and specify a custom prompt:
   ```bash
   OMP_NUM_THREADS=<number of physical cores> numactl -m 0 -C all \ 
   python quantize_transformer_models_with_itrex.py \
    --model_name "Intel/neural-chat-7b-v3-1" \
    --no_neural_speed \
    --quantize "int4" \
    --prompt "Custom prompt text goes here" \
    --max_new_tokens 50
   ```

3. Use a [Llama2-7B GGUF model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) from HuggingFace model hub . When using GGUF model, `tokenizer_name` and `model_name` arguments are required and Neural Speed is enable by default. **Note**: You will need to request access for Llama2 on HuggingFace to run the below command.
   ```bash
   OMP_NUM_THREADS=<number of physical cores> numactl -m 0 -C all \ 
   python quantize_transformer_models_with_itrex.py \
    --model_name "TheBloke/Llama-2-7B-Chat-GGUF" \
    --model_file "llama-2-7b-chat.Q4_0.gguf"\
    --tokenizer_name "meta-llama/Llama-2-7b-chat-hf"  \
    --prompt "Custom prompt text goes here" \
    --max_new_tokens 50
   ```

### Additional Notes

- Ensure that you follow the provided instructions carefully to execute the project successfully.
- Make sure to adjust the command parameters based on your specific system configuration and requirements.


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
   quantize_transformer_models_with_itrex.ipynb
   ````
6. Change your Jupyter Notebook kernel to corresponding environment.
7. Run every cell in the Notebook in sequence.

### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample shows statistics for model size and memory comsumption, before and after quantization.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).