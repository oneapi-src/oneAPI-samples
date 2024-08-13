# Modin* Get Started Sample

The `Modin Getting Started` sample demonstrates how to use distributed Pandas using the Modin package. 

| Area                  | Description
| :---                  | :---
| Category              | Getting Started
| What you will learn   | Basic Modin programming model for Intel processors
| Time to complete      | 5 to 8 minutes

## Purpose

Modin uses Ray or Dask to provide a method to speed up your Pandas notebooks, scripts, and libraries. Unlike other distributed DataFrame libraries, Modin provides integration and compatibility with existing Pandas code.

In this sample, you will run Modin-accelerated Pandas functions and note the performance gain when compared to "stock" (or standard) Pandas functions.

## Prerequisites

| Optimized for                     | Description
| :---                              | :---
| OS                                | Ubuntu* 18.04 (or newer)
| Hardware                          | Intel® Atom® processors <br> Intel® Core™ processor family <br> Intel® Xeon® processor family <br> Intel® Xeon® Scalable Performance processor family
| Software                          | Modin
> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

## Key Implementation Details

This get started sample code is implemented for CPU using the Python language. The example assumes you have Pandas and Modin installed inside a conda environment.

## Environment Setup
You will need to download and install the following toolkits, tools, and components to use the sample.

**1. Get AI Tools**

Required AI Tools: Modin

If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

>**Note**: If Docker option is chosen in AI Tools Selector, refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

**2. (Offline Installer) Activate the AI Tools bundle base environment**
<!-- this step is from AI Tools GSG, please don't modify unless GSG is updated -->
If the default path is used during the installation of AI Tools:
```
source $HOME/intel/oneapi/intelpython/bin/activate
```
If a non-default path is used:
```
source <custom_path>/bin/activate
```

**3. (Offline Installer) Activate relevant Conda environment**
<!-- specify relevant conda environment name in Offline Installer for this sample -->
```
conda activate modin  
``` 

**4. Clone the GitHub repository**
<!-- for oneapi-samples: git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/<samples-folder>/<individual-sample-folder> -->
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/Modin_GettingStarted
```

**5. Install dependencies**
<!-- It is required to have requirement.txt file in sample dir. It should list additional libraries, such as matplotlib, ipykernel etc. -->
>**Note**: Before running the following commands, make sure your Conda/Python environment with AI Tools installed is activated

```
pip install -r requirements.txt
pip install notebook
``` 
For Jupyter Notebook, refer to [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.

## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/INC-Quantization-Sample-for-PyTorch#environment-setup) is completed.

Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip) 
* [Docker](#docker)


### AI Tools Offline Installer (Validated)  

**1. Register Conda kernel to Jupyter Notebook kernel**

If the default path is used during the installation of AI Tools:
```
$HOME/intel/oneapi/intelpython/envs/modin/bin/python -m ipykernel install --user --name=modin
```
If a non-default path is used:
```
<custom_path>/bin/python -m ipykernel install --user --name=modin
```
**2. Launch Jupyter Notebook** 
<!-- add other flags to jupyter notebook command if needed, such as port 8888 or allow-root -->
```
jupyter notebook --ip=0.0.0.0
```
**3. Follow the instructions to open the URL with the token in your browser**

**4. Select the Notebook**
<!-- add sample file name -->
```
Modin_GettingStarted.ipynb
```
**5. Change the kernel to `modin`**
  <!-- specify relevant kernel name(s), for example `pytorch` -->
**6. Run every cell in the Notebook in sequence**

### Conda/PIP
> **Note**: Before running the instructions below, make sure your Conda/Python environment with AI Tools installed is activated

**1. Register Conda/Python kernel to Jupyter Notebook kernel** 
<!-- keep placeholders in this step, user could use any name for Conda/PIP env -->
For Conda:
```
<CONDA_PATH_TO_ENV>/bin/python -m ipykernel install --user --name=<your-env-name>
```
To know <CONDA_PATH_TO_ENV>, run `conda env list` and find your Conda environment path.

For PIP:
```
python -m ipykernel install --user --name=<your-env-name>
```
**2. Launch Jupyter Notebook**
<!-- add other flags to jupyter notebook command if needed, such as port 8888 or allow-root --> 
```
jupyter notebook --ip=0.0.0.0
```
**3. Follow the instructions to open the URL with the token in your browser**

**4. Select the Notebook**
<!-- add sample file name -->
```
Modin_GettingStarted.ipynb
```
**5. Change the kernel to `<your-env-name>`**
<!-- leave <your-env-name> as a placeholder as user could choose any name for the env -->

**6. Run every cell in the Notebook in sequence**

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.


### Expected Output

The expected cell output is shown in the `Modin_GettingStarted.ipynb` Notebook.

## Related Samples

* [Modin Vs. Pandas Performance](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/Modin_Vs_Pandas)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
