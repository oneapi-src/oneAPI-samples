# Job Recommendation System: End-to-End Deep Learning Workload
<!-- Do not use backticks (`) to highlight parts of the title. -->

This sample illustrates the use of Intel® Extension for TensorFlow* to build and run an end-to-end AI workload on the example of the job recommendation system.

| Property            | Description
|:---                 |:---
| Category            | Reference Designs and End to End
| What you will learn | How to use Intel® Extension for TensorFlow* to build end to end AI workload?
| Time to complete    | 30 minutes

## Purpose

This code sample show end-to-end Deep Learning workload in the example of job recommendation system. It consists of four main parts:

1. Data exploration and visualization - showing what the dataset is looking like, what are some of the main features and what is a data distribution in it.
2. Data cleaning and pre-processing - removal of duplicates, explanation all necessary steps for text pre-processing.
3. Fraud job postings removal - finding which of the job posting are fake using LSTM DNN and filtering them.
4. Job recommendation - calculation and providing top-n job descriptions similar to the chosen one.

## Prerequisites

| Optimized for       | Description
| :---                | :---
| OS                  | Linux, Ubuntu* 20.04
| Hardware            | GPU
| Software            | Intel® Extension for TensorFlow*
> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).
<!-- for migrated samples - modify the note above to provide information on samples validation and preferred installation option -->

## Key Implementation Details

This sample creates Deep Neural Networ to fake job postings detections using Intel® Extension for TensorFlow* LSTM layer on GPU. It also utilizes `itex.experimental_ops_override()` to automatically replace some TensorFlow operators by Custom Operators form Intel® Extension for TensorFlow*.

The sample tutorial contains one Jupyter Notebook and one Python script. You can use either.

## Environment Setup
You will need to download and install the following toolkits, tools, and components to use the sample.
<!-- Use numbered steps instead of subheadings -->

**1. Get AI Tools**

Required AI Tools: <Intel® Extension for TensorFlow* - GPU><!-- List specific AI Tools that needs to be installed before running this sample --> 

If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/frameworks-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.

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
conda activate tensorflow-gpu 
``` 

**4. Clone the GitHub repository**
<!-- for oneapi-samples: git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/<samples-folder>/<individual-sample-folder> -->
<!-- for migrated samples - provide git clone command for individual repo and cd to sample dir --> 
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/End-to-end-Workloads/JobRecommendationSystem
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

Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/frameworks-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Conda/PIP](#condapip) 
* [Docker](#docker)
<!-- for migrated samples - it's acceptable to change the order of the sections based on the validated/preferred installation options. However, all 3 sections (Offline, Conda/PIP, Docker) should be present in the doc -->  
### AI Tools Offline Installer (Validated)  

**1. Register Conda kernel to Jupyter Notebook kernel**

If the default path is used during the installation of AI Tools:
```
$HOME/intel/oneapi/intelpython/envs/tensorflow-gpu/bin/python -m ipykernel install --user --name=tensorflow-gpu 
```
If a non-default path is used:
```
<custom_path>/bin/python -m ipykernel install --user --name=tensorflow-gpu
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
JobRecommendationSystem.ipynb
```
**5. Change the kernel to `tensorflow-gpu`**
  <!-- specify relevant kernel name(s), for example `pytorch` -->
**6. Run every cell in the Notebook in sequence**

### Conda/PIP
> **Note**: Before running the instructions below, make sure your Conda/Python environment with AI Tools installed is activated

**1. Register Conda/Python kernel to Jupyter Notebook kernel** 
<!-- keep placeholders in this step, user could use any name for Conda/PIP env -->
For Conda:
```
<CONDA_PATH_TO_ENV>/bin/python -m ipykernel install --user --name=tensorflow-gpu
```
To know <CONDA_PATH_TO_ENV>, run `conda env list` and find your Conda environment path.

For PIP:
```
python -m ipykernel install --user --name=tensorflow-gpu
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
JobRecommendationSystem.ipynb
```
**5. Change the kernel to `<your-env-name>`**
<!-- leave <your-env-name> as a placeholder as user could choose any name for the env -->

**6. Run every cell in the Notebook in sequence**

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

<!-- Remove Intel® DevCloud section or other outdated sections -->

## Example Output
 
 If successful, the sample displays [CODE_SAMPLE_COMPLETED_SUCCESSFULLY]. Additionally, the sample shows multiple diagram explaining dataset, the training progress for fraud job posting detection and top job recommendations.

## Related Samples

<!--List other AI samples targeting similar use-cases or using the same AI Tools.-->
* [Intel Extension For TensorFlow Getting Started Sample](https://github.com/oneapi-src/oneAPI-samples/blob/development/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_TensorFlow_GettingStarted/README.md)
* [Leveraging Intel Extension for TensorFlow with LSTM for Text Generation Sample](https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/Features-and-Functionality/IntelTensorFlow_TextGeneration_with_LSTM/README.md)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
