# End-to-end Machine Learning Workload: `Census` Sample

This sample code illustrates how to use Intel® Distribution of Modin for ETL operations and ridge regression algorithm from the Intel® oneAPI Data Analytics Library (oneDAL) accelerated scikit-learn library to build and run an end to end machine learning workload. Both Intel Distribution of Modin and oneDAL accelerated scikit-learn libraries are available together in [Intel&reg; oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html). This sample code demonstrates how to seamlessly run the end-to-end census workload using the toolkit, without any external dependencies.

| Optimized for                     | Description
| :---                              | :---
	@@ -27,17 +27,21 @@ This end-to-end workload sample code is implemented for CPU using the Python lan
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

## Running Samples on the Intel&reg; DevCloud
If you are running this sample on the Intel&reg; DevCloud, skip the Pre-requirements and go to the [Activate Conda Environment](#activate-conda) section.

## Building Intel® Distribution of Modin and Intel® oneAPI Data Analytics Library (oneDAL) for CPU to build and run end-to-end workload

### Pre-requirements (Local or Remote Host Installation)
Intel® Distribution of Modin and Intel® oneAPI Data Analytics Library (oneDAL) is ready for use once you finish the Intel AI Analytics Toolkit installation with the Conda Package Manager.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi), and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) for installation steps and scripts.

### Activate conda environment With Root Access<a name="activate-conda"></a>

In the Linux shell, navigate to your oneapi installation path, typically `/opt/intel/oneapi/` when installed as root or sudo, and `~/intel/oneapi/` when not installed as a super user. 

Activate the conda environment with the following command:

	@@ -48,7 +52,7 @@ source activate intel-aikit-modin

### Activate conda environment Without Root Access (Optional)

By default, the Intel oneAPI AI Analytics toolkit is installed in the `oneapi` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### Linux
```
	@@ -62,9 +66,9 @@ conda activate intel-aikit-modin
```


### Install Jupyter Notebook*

Launch Jupyter Notebook in the directory housing the code example.

```
conda install jupyter nb_conda_kernels
	@@ -76,7 +80,7 @@ pip install jupyter

### Install wget package

Install wget package to retrieve the Census dataset using HTTPS.

```
pip install wget
	@@ -85,7 +89,7 @@ pip install wget
#### View in Jupyter Notebook


Launch Jupyter Notebook in the directory housing the code example.

```
jupyter notebook
	@@ -112,3 +116,16 @@ Run the Program
##### Expected Printed Output:
Expected Cell Output shown for census_modin.ipynb:
![Output](Expected_output.jpg "Expected output for Jupyter Notebook")


### Request a Compute Node
In order to run on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
 This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

<!---Mark each compatible Node in BOLD-->
| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| GPU               | qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh           |
| CPU               | qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh          |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |
