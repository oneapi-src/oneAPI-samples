# End-to-end Machine Learning Workload: `Census` Sample

This sample code illustrates how to use Intel® Distribution of Modin* for ETL operations and ridge regression algorithm from the Intel® extension of scikit-learn library to build and run an end to end machine learning workload. Both Intel Distribution of Modin* and  Intel® Extension for Scikit-learn libraries are available together in [Intel&reg; oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html). This sample code demonstrates how to seamlessly run the end-to-end census workload using the toolkit, without any external dependencies.

| Optimized for                     | Description
| :---                              | :---
| OS                                | 64-bit Linux: Ubuntu 18.04 or higher
| Hardware                          | Intel Atom® Processors; Intel® Core™ Processor Family; Intel® Xeon® Processor Family; Intel® Xeon® Scalable processor family
| Software                          | Intel® AI Analytics Toolkit (Python version 3.7, Intel Distribution of Modin* , Ray, Intel® Extension for Scikit-Learn, NumPy)
| What you will learn               | How to use Intel Distribution of Modin* and Intel® Extension for Scikit-learn to build end to end ML workloads and gain performance.
| Time to complete                  | 15-18 minutes

## Purpose
Intel Distribution of Modin* uses Ray to provide an effortless way to speed up your Pandas notebooks, scripts and libraries. Unlike other distributed DataFrame libraries, Intel Distribution of Modin* provides seamless integration and compatibility with existing Pandas code. Intel(R) Extension for Scikit-learn dynamically patches scikit-learn estimators to use Intel(R) oneAPI Data Analytics Library as the underlying solver, while getting the same solution faster.

#### Model and dataset
In this sample, you will use Intel Distribution of Modin* to ingest and process U.S. census data from 1970 to 2010 in order to build a ridge regression based model to find the relation between education and the total income earned in the US.
Data transformation stage normalizes the income to the yearly inflation, balances the data such that each year has a similar number of data points, and extracts the features from the transformed dataset. The feature vectors are fed into the ridge regression model to predict the education of each sample.

Dataset is from IPUMS USA, University of Minnesota, [www.ipums.org](https://ipums.org/) (Steven Ruggles, Sarah Flood, Ronald Goeken, Josiah Grover, Erin Meyer, Jose Pacas and Matthew Sobek. IPUMS USA: Version 10.0 [dataset]. Minneapolis, MN: IPUMS, 2020. https://doi.org/10.18128/D010.V10.0)

## Key Implementation Details
This end-to-end workload sample code is implemented for CPU using the Python language.  With the installation of Intel AI Analytics Toolkit, the conda environment is prepared with Python version 3.7, Intel Distribution of Modin* , Ray, Intel® Extension for Scikit-Learn, NumPy following which the sample code can be directly run using the underlying steps in this README. 

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

## Building Intel Distribution of Modin* and Intel® Extension for Scikit-learn for CPU to build and run end-to-end workload
Intel Distribution of Modin* and Intel® Extension for Scikit-learn is ready for use once you finish the Intel AI Analytics Toolkit installation with the Conda Package Manager.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi), and the Intel® oneAPI Toolkit [Installation Guide](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/conda/install-intel-ai-analytics-toolkit-via-conda.html) for conda environment setup and installation steps.

### Activate conda environment

To install the Intel® Distribution of Modin* python environment, use the following command:
#### Linux
```
conda create -y -n intel-aikit-modin intel-aikit-modin -c intel
```
Then activate your conda environment with the following command:
```
conda activate intel-aikit-modin
```

Additionally, install the following in the conda environment

### Install Jupyter Notebook
Needed to launch Jupyter Notebook in the directory housing the code example
```
conda install jupyter nb_conda_kernels
```

### opencensus
```
pip install opencensus
```

#### View in Jupyter Notebook
Launch Jupyter Notebook in the directory housing the code example
```
jupyter notebook
```
## Running the end-to-end code sample
### Run as Jupyter Notebook
Open .ipynb file and run cells in Jupyter Notebook using the "Run" button. Alternatively, the entire workbook can be run using the "Restart kernel and re-run whole notebook" button. (see image below using "census modin" sample)
![Click the Run Button in the Jupyter Notebook](Running_Jupyter_notebook.jpg "Run Button on Jupyter Notebook")

### Run as Python File
Open notebook in Jupyter and download as python file (see image using "census modin" sample)
![Download as python file in the Jupyter Notebook](Running_Jupyter_notebook_as_Python.jpg "Download as python file in the Jupyter Notebook")
Run the Program
`python census_modin.py`
##### Expected Printed Output:
Expected Cell Output shown for census_modin.ipynb:
![Output](Expected_output.jpg "Expected output for Jupyter Notebook")
