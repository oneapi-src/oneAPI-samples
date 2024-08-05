# oneAPI Deep Neural Network Library (oneDNN)

oneAPI Deep Neural Network Library (oneDNN) is an open-source performance
library for deep learning applications. The library includes basic building
blocks for neural networks optimized for Intel Architecture Processors
and Intel Processor Graphics. oneDNN is intended for deep learning
applications and framework developers interested in improving application
performance on Intel CPUs and GPUs.

You can find library source code and code used by these samples at [oneDNN GitHub repository](https://github.com/oneapi-src/oneDNN).

See the [Intel® oneAPI Deep Neural Network Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html) product page for details on getting oneDNN.

# oneDNN Samples

| Type      | Name                                             | Description
| --------- | ------------------------------------------------ | -
| Component | [getting_started](getting_started)               | A C++ sample demonstrating basics of oneDNN programming model.
| Component | [tutorials](tutorials)                           | Hands-on Jupyter notebook tutorials among different topics.

# Using Samples in Intel® DevCloud for oneAPI

You can use oneDNN samples in
[Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get-started/)
the environment in the following ways:

* Log in to a DevCloud system via SSH and
  * use `git clone` to get a full copy of samples repository, or
  * use the `oneapi-cli` tool to download specific sample.
* Launch a JupyterLab server and run Jupyter Notebooks from your web browser.

# Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).