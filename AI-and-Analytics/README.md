# Intel® oneAPI AI Analytics Toolkit (AI Kit)

The Intel® oneAPI AI Analytics Toolkit (AI Kit) gives data scientists, AI developers, and researchers familiar Python* tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. The components are built using oneAPI libraries for low-level compute optimizations. This toolkit maximizes performance from preprocessing through machine learning, and provides interoperability for efficient model development.

You can find more information at [ AI Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# AI Samples

| Type      | Folder                                             | Description
| --------- | ------------------------------------------------ | -
| Component | [Getting-Started-Samples](Getting-Started-Samples)               | Getting Started Samples for components in AI Kit.
| Component & Segment | [Features-and-Functionality](Features-and-Functionality) | Demonstrate features from components like Int8 inference in Model Zoo.
| Reference | [End-to-end-Workloads](End-to-end-Workloads)                     | AI End-to-end reference workloads with real world data.

# Using Samples in Intel® DevCloud for oneAPI

## General DevCloud Usage Instructions:
You can use AI Kit samples in
the [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get-started/) environment in the following ways:
* Log in to a DevCloud system via SSH
* Launch a JupyterLab server and run Jupyter Notebooks from your web browser.   
> Please refer to [DevCloud README](DevCloudREADME.md) for more details.
## Get Code Samples
* use `git clone` to get a full copy of samples repository, or
* use the `oneapi-cli` tool to download specific sample.
> Users could refer to [the Download Samples using the oneAPI CLI Samples Browser section](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-hpc-linux/top/run-a-sample-project-using-the-command-line.html).
## How to submit a workload to a specific architecture
* check the available nodes with your DevCloud account 
```
./q -h
```
* select one of available node for your workload. 
ex: select a Cascade Lake node to run your workload
```
export TARGET_NODE=clx
```
* prepare a run script which contains all needed run commands for your workload. 
> Users could refer to [run.sh for TensorFlow Getting started sample](https://github.com/intel-ai-tce/oneAPI-samples/blob/devcloud/AI-and-Analytics/Getting-Started-Samples/IntelTensorFlow_GettingStarted/run.sh).
* submit your workload on the selected node with the run script.
```
./q ./run.sh
```
