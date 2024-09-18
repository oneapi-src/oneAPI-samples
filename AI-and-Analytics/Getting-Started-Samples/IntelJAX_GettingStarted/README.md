# `JAX Getting Started` Sample

The `JAX Getting Started` sample demonstrates how to train a JAX model and run inference on Intel® hardware. 
| Property            | Description 
|:---                 |:---
| Category            | Get Start Sample 
| What you will learn | How to start using JAX* on Intel® hardware.
| Time to complete    | 10 minutes

## Purpose

JAX is a high-performance numerical computing library that enables automatic differentiation. It provides features like just-in-time compilation and efficient parallelization for machine learning and scientific computing tasks.

This sample code shows how to get started with JAX in CPU. The sample code defines a simple neural network that trains on the MNIST dataset using JAX for parallel computations across multiple CPU cores. The network trains over multiple epochs, evaluates accuracy, and adjusts parameters using stochastic gradient descent across devices. 

## Prerequisites

| Optimized for          | Description
|:---                    |:---
| OS                     | Ubuntu* 22.0.4 and newer 
| Hardware               | Intel® Xeon® Scalable processor family
| Software               | JAX

> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

## Key Implementation Details

The example implementation involves a python file 'spmd_mnist_classifier_fromscratch.py' under the examples directory from the jax repo [(https://github.com/google/jax/)].
It implements a simple neural network's training and inference for mnist images. The images are downloaded to a temporary directory when the example is run first. 
- **init_random_params** initializes the neural network weights and biases for each layer.
- **predict** computes the forward pass of the network, applying weights, biases, and activations to inputs.
- **loss** calculates the cross-entropy loss between predictions and target labels.
- **spmd_update** performs parallel gradient updates across multiple devices using JAX’s pmap and lax.psum.
- **accuracy** computes the accuracy of the model by predicting the class of each input in the batch and comparing it to the true target class. It uses the *jnp.argmax* function to find the predicted class and then computes the mean of correct predictions.
- **data_stream** function generates batches of shuffled training data. It reshapes the data so that it can be split across multiple cores, ensuring that the batch size is divisible by the number of cores for parallel processing.
- **training loop** trains the model for a set number of epochs, updating parameters and printing training/test accuracy after each epoch. The parameters are replicated across devices and updated in parallel using spmd_update. After each epoch, the model’s accuracy is evaluated on both training and test data using accuracy.

## Environment Setup

You will need to download and install the following toolkits, tools, and components to use the sample.

**1. Get Intel® AI Tools**

Required AI Tools: 'JAX' 
<br>If you have not already, select and install these Tools via [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html). AI and Analytics samples are validated on AI Tools Offline Installer. It is recommended to select Offline Installer option in AI Tools Selector.<br>
please see the [supported versions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html).

>**Note**: If Docker option is chosen in AI Tools Selector, refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

**2. (Offline Installer) Activate the AI Tools bundle base environment**

If the default path is used during the installation of AI Tools:
```
source $HOME/intel/oneapi/intelpython/bin/activate
```
If a non-default path is used:
```
source <custom_path>/bin/activate
```
 
**3. (Offline Installer) Activate relevant Conda environment**

For the system with Intel CPU:
```
conda activate jax
``` 

**4. Clone the GitHub repository**
``` 
git clone https://github.com/google/jax.git
cd jax
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
## Run the Sample

>**Note**: Before running the sample, make sure Environment Setup is completed.
Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)/Conda/PIP](#ai-tools-offline-installer-validatedcondapip)
* [Docker](#docker)
### AI Tools Offline Installer (Validated)/Conda/PIP
```
 python examples/spmd_mnist_classifier_fromscratch.py
```
### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.
## Example Output
1. With the initial run, you should see results similar to the following:

```
downloaded https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz to /tmp/jax_example_data/
downloaded https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz to /tmp/jax_example_data/
downloaded https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz to /tmp/jax_example_data/
downloaded https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz to /tmp/jax_example_data/
Epoch 0 in 2.71 sec
Training set accuracy 0.7381166815757751
Test set accuracy 0.7516999840736389
Epoch 1 in 2.35 sec
Training set accuracy 0.81454998254776
Test set accuracy 0.8277999758720398
Epoch 2 in 2.33 sec
Training set accuracy 0.8448166847229004
Test set accuracy 0.8568999767303467
Epoch 3 in 2.34 sec
Training set accuracy 0.8626833558082581
Test set accuracy 0.8715999722480774
Epoch 4 in 2.30 sec
Training set accuracy 0.8752999901771545
Test set accuracy 0.8816999793052673
Epoch 5 in 2.33 sec
Training set accuracy 0.8839333653450012
Test set accuracy 0.8899999856948853
Epoch 6 in 2.37 sec
Training set accuracy 0.8908833265304565
Test set accuracy 0.8944999575614929
Epoch 7 in 2.31 sec
Training set accuracy 0.8964999914169312
Test set accuracy 0.8986999988555908
Epoch 8 in 2.28 sec
Training set accuracy 0.9016000032424927
Test set accuracy 0.9034000039100647
Epoch 9 in 2.31 sec
Training set accuracy 0.9060333371162415
Test set accuracy 0.9059999585151672
```

2. Troubleshooting

   If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
