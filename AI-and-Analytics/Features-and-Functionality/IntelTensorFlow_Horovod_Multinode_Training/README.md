# Distributed TensorFlow with Horovod Sample
Today's modern computer systems are becoming heavily distributed and it is important to capitalise on scaling techniques to maximise the efficiency and performance of training of neural networks, which is a resource intensive process.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | Intel® Xeon® Scalable processor family or newer
| Software                          | Intel® oneAPI AI Analytics Toolkit
| What you will learn               | How to get scale out (distribute) the training of a model on multiple compute nodes
| Time to complete                  | 10 minutes

## Purpose
This sample code shows how to get started with scaling out the training of a neural network in TensorFlow on multiple compute nodes in a cluster. The sample uses  [Horovod](https://github.com/horovod/horovod)*, which is a distributed deep learning training framework, to  facilitate the task of distributing the workload. Horovod's  core principles are based on MPI concepts such as size, rank, local rank, allreduce, allgather and, broadcast.

Intel-optimized Tensorflow is available as part of Intel® AI Analytics Toolkit. For more information on the optimizations as well as performance data, see this blog post [TensorFlow* Optimizations on Modern Intel® Architecture](https://software.intel.com/content/www/us/en/develop/articles/tensorflow-optimizations-on-modern-intel-architecture.html).

## Key implementation details

 - The training dataset is comes from Keras*'s build-in dataset.
 - The dataset will be split based on the number of MPI ranks 
 - We load Horovod and initialize the framework using `hvd.init()`
 - We then wrap the optimzier around Horovod's distributed optimizer with `opt = hvd.DistributedOptimizer(opt)`
 - The appropriate hooks are configured and we make sure that only rank 0 writes the checkpoints
    
Runtime settings for `OMP_NUM_THREADS`, `KMP_AFFINITY`, and `Inter/Intra-op` Threads are set within the script. You can read more about these settings in this dedicated document: [Maximize TensorFlow Performance on CPU: Considerations and Recommendations for Inference Workloads](https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference) 
    
## License  
This code sample is licensed under MIT license.

## Build and Run the Sample

### Running Samples In DevCloud (Optional)
If running a sample in the Intel DevCloud, please follow the below steps to build the python environment. Also remember that you must specify the compute node (CPU, GPU, FPGA) as well whether to run in batch or interactive mode. For more information see the Intel® oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/) 

### Pre-requirement

TensorFlow is ready for use once you finish the Intel AI Analytics Toolkit installation, and have run post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation, and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.


### On a Linux* System
#### Activate conda environment With Root Access

Navigate in linux shell to your oneapi installation path, typically `/opt/intel/oneapi`. Activate the conda environment with the following command:

```
source /opt/intel/oneapi/setvars.sh
source activate tensorflow
```

#### Activate conda environment Without Root Access (Optional)

By default, the Intel AI Analytics toolkit is installed in the `/opt/intel/oneapi` folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

```
conda create --name user_tensorflow --clone tensorflow
```

Then activate your conda environment with the following command:

```
source activate user_tensorflow
```

## Running the Sample

Before you proceed with running the sample, you will need to install the 3rd-party [Horovod](https://github.com/horovod/horovod) framework. 

After you have activated your conda environment, you may wish to execute the following commands to install `horovod`:
```
export HOROVOD_WITHOUT_MPI=1 #Optional, in case you encouter MPI-related install issues
pip install horovod
```

To the script on one machine without invoking Horovod, type the following command in the terminal with Python installed:
```
    python TensorFlow_Multinode_Training_with_Horovod.py
```

To run the script with Horovod, we invoke MPI:
```
    horovodrun -np 2 TensorFlow_Multinode_Training_with_Horovod.py
```

In the example above, we run the script on two MPI threads but on the same node. To use multiple nodes, we will pass the `-hosts` flag, where host1 and host2 are the hostname of two nodes on your cluster. 

Example:

```
    horovodrun -n 2 -H host1,host2 TensorFlow_Multinode_Training_with_Horovod.py
```


### Example of Output
With successful execution, it will print out the following results:

```
[...]
I0930 19:50:23.496505 140411946298240 basic_session_run_hooks.py:606] Saving checkpoints for 0 into ./checkpoints/model.ckpt.
INFO:tensorflow:loss = 2.2964332, step = 1
I0930 19:50:25.164514 140410195934080 basic_session_run_hooks.py:262] loss = 2.2964332, step = 1
INFO:tensorflow:loss = 2.2851412, step = 1
I0930 19:50:25.186133 140411946298240 basic_session_run_hooks.py:262] loss = 2.2851412, step = 1
INFO:tensorflow:loss = 1.1958275, step = 101 (19.356 sec)
I0930 19:50:44.521067 140410195934080 basic_session_run_hooks.py:260] loss = 1.1958275, step = 101 (19.356 sec)
INFO:tensorflow:global_step/sec: 5.16882
[...]
============================
Number of tasks:  2
Total time is: 96.9508
```


