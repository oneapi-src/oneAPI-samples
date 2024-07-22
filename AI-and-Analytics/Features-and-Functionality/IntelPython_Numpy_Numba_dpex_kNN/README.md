# `Intel® Python: NumPy vs numba-dpex` Sample

The `Intel® Python: NumPy vs numba-dpex` sample shows how to achieve the same accuracy of the k-NN model classification while using NumPy*, Numba*, and Data Parallel Extension for Numba* (numba-dpex).

| Property                | Description
| :---                    | :---
| Category                | Code Optimization
| What you will learn     | How to program using the Data Parallel Extension for Numba* (numba-dpex)
| Time to complete        | 5 minutes

>**Note**: This sample is validated on Intel® Distribution for Python* Offline Installer and AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

## Purpose

In this sample, you will run a k-nearest neighbors algorithm using 3 different Intel® Distribution for Python* libraries: NumPy, Numba, and numba-dpex. You will learn how to use k-NN model and how to optimize it by numba-dpex operations without sacrificing accuracy.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04
| Hardware                | CPU
| Software                | Intel® Distribution for Python*

## Key Implementation Details

This sample code is implemented for the CPU using Python. The sample assumes you have numba-dpex installed inside a Conda environment, similar to what is installed with the Intel® Distribution for Python*.

The sample tutorial contains one Jupyter Notebook and one Python script. You can use either.

## Environment Setup
You will need to download and install the following toolkits to use the sample.
<!-- Use numbered steps instead of subheadings -->

**1. Get Intel® Distribution for Python***

If you have not already, install Intel® Distribution for Python* via [Installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-python-download.html?operatingsystem=linux&linux-install-type=offline).

**2. Activate the Intel® Distribution for Python\* base environment**
<!-- this step is from AI Tools GSG, please don't modify unless GSG is updated -->
If the default path is used during the installation of Intel® Distribution for Python*:
```
source $HOME/intelpython3/bin/activate
```
If a non-default path is used:
```
source <custom_path>/bin/activate
```

**3. Clone the GitHub repository**
<!-- for oneapi-samples: git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/<samples-folder>/<individual-sample-folder> -->
<!-- for migrated samples - provide git clone command for individual repo and cd to sample dir --> 
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Features-and-Functionality/IntelPython_Numpy_Numba_dpex_kNN
```

**4. Install dependencies**
<!-- It is required to have requirement.txt file in sample dir. It should list additional libraries, such as matplotlib, ipykernel etc. -->
>**Note**: Before running the following commands, make sure your Conda environment is activated

```
pip install ipykernel
pip install notebook
``` 
For Jupyter Notebook, refer to [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.

## Run the Sample
>**Note**: Before running the sample, make sure [Environment Setup](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPython_Numpy_Numba_dpex_kNN#environment-setup) is completed.

### Intel® Distribution for Python* Offline Installer (Validated)

**1. Register Conda kernel to Jupyter Notebook kernel**

If the default path is used during the installation of Intel® Distribution for Python*:
```
$HOME/intelpython3/bin/python -m ipykernel install --user --name=base
```
If a non-default path is used:
```
<custom_path>/bin/python -m ipykernel install --user --name=base
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
IntelPython_Numpy_Numba_dpex_kNN.ipynb
```
**5. Change the kernel to `base`**
  <!-- specify relevant kernel name(s), for example `pytorch` -->
**6. Run every cell in the Notebook in sequence**

## Example Output

```
Numpy accuracy: 0.7222222222222222

Numba accuracy: 0.7222222222222222

Numba_dpex accuracy 0.7222222222222222

[CODE_SAMPLE_COMPLETED_SUCCESFULLY]
```

## Related Samples

* [Get Started with the Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/articles/technical/get-started-with-intel-distribution-for-python.html)
* [`Genetic Algorithms on GPU using Intel® Distribution for Python* numba-dpex` Sample](https://github.com/oneapi-src/AI-and-Analytics/Features-and-Functionality/IntelPython_GPU_numba-dpex_Genetic_Algorithm/README.md)
## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
