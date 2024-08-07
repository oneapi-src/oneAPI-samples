# `Interactive chat based on DialoGPT model using Intel® Extension for PyTorch (IPEX) Quantization` Sample

The `Interactive chat based on DialoGPT model using Intel® Extension for PyTorch (IPEX) Quantization` sample demonstrates how to create interactive chat based on pre-trained DialoGPT model and add the Intel® Extension for PyTorch (IPEX) quantization to it.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to create interactive chat and add INT8 dynamic quantization form Intel® Extension for PyTorch (IPEX)
| Time to complete      | 10 minutes
| Category              | Concepts and Functionality

The Intel® Extension for PyTorch (IPEX) extends PyTorch* with optimizations for extra performance boost on Intel® hardware. While most of the optimizations will be included in future PyTorch* releases, the extension delivers up-to-date features and optimizations for PyTorch on Intel® hardware. For example, newer optimizations include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

## Purpose

This sample shows how to create interactive chat based on the pre-trained DialoGPT model from HuggingFace and how to add INT8 dynamic quantization to it. The Intel® Extension for PyTorch (IPEX) gives users the ability to speed up operations on processors with INT8 data format and specialized computer instructions. The INT8 data format uses quarter the bit width of floating-point-32 (FP32), lowering the amount of memory needed and execution time to process with minimum to zero accuracy loss.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04 or newer
| Hardware                | Intel® Xeon® Scalable Processor family
| Software                | Intel® Extension for PyTorch (IPEX)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See *[Intel® DevCloud for oneAPI](https://DevCloud.intel.com/oneapi/get_started/)* for information.

## Key Implementation Details

This code sample implements interactive chat based on DialoGPT pre-trained model and quantizes it using Intel® Extension for PyTorch (IPEX).

The sample tutorial contains one Jupyter Notebook and a Python script. You can use either.

### Jupyter Notebook

| Notebook                         | Description
|:---                              |:---
|`IntelPytorch_Interactive_Chat_Quantization.ipynb` | Performs chat creation with Intel® Extension for PyTorch (IPEX) quantization and provides interface for interactions in Jupyter Notebook.

### Python Scripts

| Script                        | Description
|:---                           |:---
|`IntelPytorch_Interactive_Chat_Quantization.py` | The script performs chat creation with Intel® Extension for PyTorch (IPEX) quantization and provides simple interactions based on prepared input.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Interactive chat based on DialoGPT model using Intel® Extension for PyTorch (IPEX) Quantization` Sample

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
   IntelPytorch_Interactive_Chat_Quantization.ipynb
   ```
5. Change your Jupyter Notebook kernel to **PyTorch (AI kit)**.
6. Run every cell in the Notebook in sequence.

#### Running on the Command Line (Optional)

1. Change to the sample directory.
2. Run the script.
   ```
   python IntelPytorch_Interactive_Chat_Quantization.py < input.txt
   ```

### Run the `Interactive chat based on DialoGPT model using Intel® Extension for PyTorch (IPEX) Quantization` Sample on Intel® DevCloud

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
   IntelPytorch_Interactive_Chat_Quantization.ipynb
   ````
6. Change the kernel to **PyTorch (AI kit)**.
7. Run every cell in the Notebook in sequence.

### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

```
Loading model...
Quantization in progress...
>> You: Hello! How are you?
DialoGPT: Great and you?
>> You: I am good!
DialoGPT: Well good!
>> You: Can you go to the cinema today?
DialoGPT: Of course I can!
>> You: What movie would you like to see?
DialoGPT: Can you pick out the movies that aren't in english?
>> You: Ok, see you at cinema! Bye!
DialoGPT: See ya!
Inference with FP32
Loading model...
Warmup...
Inference...
Inference with Dynamic INT8
Loading model...
Quantization in progress...
Warmup...
Inference...
[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
