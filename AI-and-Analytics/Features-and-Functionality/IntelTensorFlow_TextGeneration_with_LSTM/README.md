# `Leveraging Intel Extension for TensorFlow with LSTM for Text Generation` Sample

The `Leveraging Intel Extension for TensorFlow with LSTM for Text Generation` sample demonstrates how to train your model for text generation with LSTM (Long short-term Memory) faster by using Intel Extension for TensorFlow's LSTM training layer on Intel platform. A model trained for text generation can be adopted later to follow specific instructions. Some examples include various chat assistants, code generators, stories generation and many more.

To train the model for text generation, a concept of memory needs to be included in the neural network. Those neural network are called RNN (Recurrent Neural Network). In this sample, a LSTM type of RNN will be used. A LSTM (Long Short-term Memory) Neural Network is just another kind of Artificial Neural Network, containing LSTM cells as neurons in some of its layers. Much like Convolutional Layers help a Neural Network learn about image features, LSTM cells help the Network learn about temporal data, something which other Machine Learning models traditionally struggled with.

| Area                  | Description
|:---                   |:---
| What you will learn   | Train your model for text generation with LSTM on Intel's GPU
| Time to complete      | 15-20 minutes (without model training)
| Category              | Concepts and Functionality

## Purpose

This sample shows how to train the model for text generation using LSTM on Intel's GPU. It will also highlight the key parts required for transitioning the existing script for model training with LSTM to Intel hardware. Leveraging Intel Extenstion for TensorFlow and its LSTM layer with Intel's GPU will provide faster training time and less GPU memory consumption.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 22.04 (or newer)
| Hardware                | Intel® Arc™, Data and Max series GPU
| Software                | Intel® AI Tools (AI Kit)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Tools (AI Kit)**

  You can get the AI Tools from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

- **Additional Packages**

  You will need to install the additional packages in requirements.txt.

  ```
  pip install -r requirements.txt
  ```

## Key Implementation Details

This code sample implements text generation model training with LSTM.

The sample tutorial contains one Jupyter Notebook and a Python script. You can use either.

### Jupyter Notebook

| Notebook                                                 | Description
|:---                                                      |:---
|`TextGenerationModelTraining.ipynb`                       | Provides interface for interactions in Jupyter Notebook.

### Python Scripts

| Script                                                   | Description
|:---                                                      |:---
|`TextGenerationModelTraining.py`                          | The script performs model training.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Leveraging Intel Extension for TensorFlow with LSTM for Text Generation` Sample

### On Linux*

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/intelpython/bin/activate`
> - For private installations: `source $HOME/intel/oneapi/intelpython/bin/activate`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

#### Activate Conda

1. Activate the Conda environment.
   ```
   conda activate tensorflow-gpu
   ```
2. Activate Conda environment without Root access (Optional).

   By default, the AI Tools is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

   ```
   conda create --name user_tensorflow-gpu --clone tensorflow-gpu
   conda activate user_tensorflow-gpu
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
   TextGenerationModelTraining.ipynb
   ```
5. Change your Jupyter Notebook kernel to corresponding environment.
6. Run every cell in the Notebook in sequence.

#### Running on the Command Line (Optional)

1. Change to the sample directory.
2. Run the script.
   ```
   python TextGenerationModelTraining.py
   ```

### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample shows the training progress and in the end seeded and generated text.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).