# `Genetic Algorithms on GPU using Intel® Distribution for Python numba-dpex` Sample

The `Genetic Algorithms on GPU using Intel® Distribution for Python numba-dpex` sample shows how to implement a general genetic algorithm (GA) and offload computation to a GPU using numba-dpex.

| Property                    | Description
| :---                    | :---
| Category                | Code Optimization
| What you will learn     | How to implement the genetic algorithm using the Data-parallel Extension for Numba* (numba-dpex)?
| Time to complete        | 8 minutes

>**Note**: The libraries used in this sample are available in Intel® Distribution for Python* [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html).

## Purpose

In this sample, you will create and run the general genetic algorithm and optimize it to run on GPU using the Intel® Distribution for Python* numba-dpex. You will learn what are selection, crossover, and mutation, and how to adjust those methods from general genetic algorithm to a specific optimization problem which is the Traveling Salesman Problem.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04
| Hardware                | GPU
| Software                | Intel® AI Analytics Toolkit (AI Kit)

## Key Implementation Details

This sample code is implemented for GPUs using Python. The sample assumes you have numba-dpex installed inside a Conda environment, similar to what is installed with the Intel® Distribution for Python*.

>**Note**: Read *[Get Started with the Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/articles/technical/get-started-with-intel-distribution-for-python.html)* to find out how you can achieve performance gains through Intel optimizations.

The sample tutorial contains one Jupyter Notebook and one Python script. You can use either.

## Environment Setup
You will need to download and install the following toolkits to use the sample.
<!-- Use numbered steps instead of subheadings -->

**1. Get Intel® Distribution for Python***

If you have not already, install Intel® Distribution for Python* via [Installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-python-download.html?operatingsystem=linux&linux-install-type=offline). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

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
conda activate base  
``` 

**4. Clone the GitHub repository**
<!-- for oneapi-samples: git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/<samples-folder>/<individual-sample-folder> -->
<!-- for migrated samples - provide git clone command for individual repo and cd to sample dir --> 
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Features-and-Functionality/IntelPython_GPU_numba-dpex_Genetic_Algorithm
```

**5. Install dependencies**
<!-- It is required to have requirement.txt file in sample dir. It should list additional libraries, such as matplotlib, ipykernel etc. -->
>**Note**: Before running the following commands, make sure your Conda/Python environment is activated

```
pip install -r requirements.txt
pip install notebook
``` 
For Jupyter Notebook, refer to [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.

## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/INC-Quantization-Sample-for-PyTorch#environment-setup) is completed.

Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [IDP Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip) 

### AI Tools Offline Installer (Validated)

**1. Register Conda kernel to Jupyter Notebook kernel**

If the default path is used during the installation of AI Tools:
```
$HOME/intel/oneapi/intelpython/envs/<offline-conda-env-name>/bin/python -m ipykernel install --user --name=<offline-conda-env-name>
```
If a non-default path is used:
```
<custom_path>/bin/python -m ipykernel install --user --name=<offline-conda-env-name>
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
IntelPython_GPU_numba-dpex_Genetic_Algorithm.ipynb
```
**5. Change the kernel to `base`**
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
```
jupyter notebook --ip=0.0.0.0
```
**3. Follow the instructions to open the URL with the token in your browser**

**4. Select the Notebook**
```
IntelPython_GPU_numba-dpex_Genetic_Algorithm.ipynb
```
**5. Change the kernel to `<your-env-name>`**

**6. Run every cell in the Notebook in sequence**

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]` at the end of execution. The sample will print out the runtimes and charts of relative performance with numba-dpex and without any optimizations as the baseline. Additionally, sample will print the best and worst path found in the Traveling Salesman problem.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
