# oneAPI Deep Neural Network Library (oneDNN)

oneAPI Deep Neural Network Library (oneDNN) is an open-source performance
library for deep learning applications. The library includes basic building
blocks for neural networks optimized for Intel Architecture Processors
and Intel Processor Graphics. oneDNN is intended for deep learning
applications and framework developers interested in improving application
performance on Intel CPUs and GPUs.

You can find library source code and code used by these samples at [oneDNN Github repository](https://github.com/oneapi-src/oneDNN).

## License
The code samples are licensed under MIT license.

# oneDNN Samples

| Type      | Name                                             | Description
| --------- | ------------------------------------------------ | -
| Component | [getting_started](getting_started)               | A C++ sample demonstrating basics of oneDNN programming model.
| Component | [dpcpp_interoparibility](dpcpp_interoperability) | A DPC++ example demonstrating interoperaility of oneDNN with DPC++ application code.
| Component | [simple_model](simple_model)                     | A C++ example demonstrating implmentation of simple convolutional model with oneDNN.
| Component | [tutorials](tutorials)                           | Hands-on Jupyter notebook tutorials among different topics.

# Using Samples in Intel oneAPI DevCloud

You can use oneDNN samples in
[Intel oneAPI DevCloud](https://devcloud.intel.com/oneapi/get-started/)
environment in the following ways:
* Login to a DevCloud system via SSH and
  * use `git clone` to get a full copy of samples repository, or
  * use `oneapi-cli` tool to download specific sample.
* Launch a JupyterLab server and run Jupyter Notebooks from your web browser.
