# `PointPillars` Sample
This sample performs 3D object detection and classification using data (point cloud) from a LIDAR sensor as input. The Intel® oneAPI implementation is based on the paper 'PointPillars: Fast Encoders for Object Detection from Point Clouds' [1] and the implementation in [2]. It shows how DPCPP and SYCL kernels can be used in combination with the Intel® Distribution of OpenVINO™ toolkit, both part of Intel® oneAPI.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer / Intel Xe Graphics
| Software                          | Intel® oneAPI DPC++/C++ Compiler, Intel® Distribution of OpenVINO™ toolkit
| What you will learn               | How to combine Intel® Distribution of OpenVINO™ toolkit and Intel® oneAPI Base Toolkit to offload the computation of a complex workload to one of Intel's supported accelerators (e.g., GPU or CPU)
| Time to complete                  | 30 minutes

## Purpose
PointPillars is an AI algorithm that uses LIDAR point clouds to detect and classify 3D objects in the sensor environment. For this purpose, the algorithm consists of the following steps, that are also visualized in the figure below:

![Overview](data/point_pillars_overview.png)

1. Pre-processing of the LiDAR input point cloud is performed. This is realized with the help of kernels implemented using SYCL and DPCPP.
2. An anchor grid is generated. The anchors in the grid are later used in object detection to refine detected boxes by the RegionProposalNetwork (RPN). The anchor grid generation is also implemented using SYCL and DPCPP.
3. Afterward, the pre-processed data is used by a so-called Pillar Feature Extraction (PFE) CNN to create a 2D image-like representation of the sensor environment. For the inference, this sample uses the Intel® Distribution of OpenVINO™ toolkit. The output of this CNN is a list of dense tensors (learned pillar features).
4. To convert these dense tensors into an pseudo-image, a scatter operation is performed. This operation is again realized with SYCL and DPCPP.
5. This pseudo-image is consumed by the second CNN, the so-called Region Proposal Network (RPN). The inference is performed with the help of the Intel® Distribution of OpenVINO™ toolkit. The output is an unfiltered list of possible object detections, their position, dimensions and classifications.
6. Finally, this output data (object list) is post-processed with the help of the anchors created in the 2nd step. The anchors are used to decode the object position, dimension and class. Afterwards, a Non-Maximum-Suppression (NMS) is used to filter out redundant/clutter objects. Finally, the objects are sorted according to their likelihood, and then provided as output. All of these steps are implemented as SYCL and DPCPP kernels.

By default, the application will use 'host' as the execution device for SYCL/DPCPP kernels and CPU (single-threaded) for Intel® Distribution of OpenVINO™ toolkit inferencing part. The execution device and the inferencing device are displayed in the output, along with the elapsed time of each of the five steps described above. For more details refer to section: [Execution Options for the Sample Program](#execution-options-for-the-sample-program).

## Key Implementation Details
This sample demonstrates a real-world, end-to-end example that uses a combination of Intel® oneAPI Base Toolkit (DPCPP, SYCL) and the Intel® Distribution of OpenVINO™ to solve object detection's complex task in a given environment. Hence, this sample will give you insights into the following aspects:
 - You will learn how to transfer data from a DPCPP/SYCL device/kernel to an OpenVINO-based inference task and back.
 - You will learn how to implement a device manager that allows choosing the target hardware for execution, i.e., CPU, GPU or an accelerator, at runtime in a user transparent manner. As a result, the target hardware can be chosen via a command-line argument without requiring a time-consuming re-compilation (further details on the execution are provided below)
 - You will learn how to implement oneAPI-based function kernels that can be executed on the host system, on a multi-threaded CPU or a GPU.
 - You will learn how to implement standard algorithms for AI-based object detection, for example, _Non-Maximum-Suppression_, using oneAPI.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Running Samples on the Intel&reg; DevCloud
If you are running this sample on the DevCloud, see [Running Samples on the Intel&reg; DevCloud](#run-samples-on-devcloud)

## Building the `PointPillars` Sample Program for CPU and GPU
Currently, only Linux platforms are supported. It is recommended to use Ubuntu 18.04.

### Requirements (Local or Remote Host Installation)
To build and run the PointPillars sample, the following libraries have to be installed:
1. Intel® Distribution of OpenVINO™ toolkit (at least 2021.1)
2. Intel® oneAPI Base Toolkit (at least 2021.2)
3. Boost (including `boost::program_options` and `boost::filesystem` library). For Ubuntu, you may install the libboost-all-dev package.
4. Optional: If the sample should be run on an Intel GPU, it might be necessary to upgrade the corresponding drivers. Therefore, please consult the following page: https://github.com/intel/compute-runtime/releases/

### Using Visual Studio Code*  (VS Code)

You can use VS Code extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### Build process (Local or Remote Host Installation)

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

1. Prepare the environment to be able to use the Intel® Distribution of OpenVINO™ toolkit and oneAPI
```
$ source /opt/intel/openvino_2021/bin/setupvars.sh
$ source /opt/intel/oneapi/setvars.sh
```

2. Build the program using the following `cmake` commands.
```
$ mkdir build && cd build
$ cmake ..
$ make
```
Please note that cmake will also download the ONNX models required for the two inference steps executed with the Intel® Distribution of OpenVINO™ toolkit.

If an error occurs, you can get more details by running `make` with the
`VERBOSE=1` argument:

 ``make VERBOSE=1``

 For more comprehensive
troubleshooting, use the Diagnostics Utility for Intel® oneAPI Toolkits, which
provides system checks to find missing dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


## Running the `PointPillars` Sample Program
After a successful build, the sample program can be run as follows:
```
./example.exe
```
The input data for the sample program is the example.pcd located in the /data folder. It contains an artificial point cloud from a simulated LIDAR sensor of the CARLA simulator [3]. The corresponding scene looks as follows:
![SampleScene](data/image.png)
The example.pcd file is a point cloud in ASCII using the PCL-format, which renders to (using the pcl_viewer tool):
![SamplePointCloud](data/pointcloud.png)
It is worth noting that there are three cars in this scene, one of which is very far away and thus not yet properly covered in the LIDAR scan. Then, there is a black car inside the intersection, which is also well visible in the LIDAR data, and a hidden car, right behind the black one. Hence, `PointPillars` should detect at least one car.

Successful execution of the sample program results in at least one detected object and an output similar to:
```
Starting PointPillars
   PreProcessing - 20ms
   AnchorMask - 10ms
   PFE Inference - 91ms
   Scattering - 50ms
   RPN Inference - 107ms
   Postprocessing - 13ms
Done
Execution time: 296ms

1 cars detected
Car: Probability = 0.622569 Position = (24.8561, 12.5615, -0.00771689) Length = 2.42855 Width = 3.61396
```

## Execution Options for the Sample Program
The sample program provides a few command-line options, which can be accessed using the 'help' option
```
./example.exe --help
```

Furthermore, it is possible to specify the execution device. For using multi-threaded CPU execution, please use:
```
./example.exe --cpu
```
For single-threaded execution on the host system, please use:
```
./example.exe --host
```
And to use an Intel® DG1 or integrated graphics, please use:
```
./example.exe --gpu
```
These options can also be used in combination, e.g.:
```
./example.exe --cpu --gpu --host
```

## Running Samples on the Intel&reg; DevCloud<a name="run-samples-on-devcloud"></a>

### Run in Batch Mode
This sample runs in batch mode, so you must have a script for batch processing. Once you have a script set up, continue with the next section, Request a Compute Node.

### Request a Compute Node
In order to run on the DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.
 This node information must be provided when submitting a job to run your sample in batch mode using the qsub command. When you see the qsub command in the Run section of the [Hello World instructions](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/), change the command to fit the node you are using. Nodes which are in bold indicate they are compatible with this sample:

<!---Mark each compatible Node in BOLD-->
| Node              | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| __GPU__           | __qsub -l nodes=1:gpu:ppn=2 -d . hello-world.sh__       |
| __CPU__           | __qsub -l nodes=1:xeon:ppn=2 -d . hello-world.sh__      |
| FPGA Compile Time | qsub -l nodes=1:fpga\_compile:ppn=2 -d . hello-world.sh |
| FPGA Runtime      | qsub -l nodes=1:fpga\_runtime:ppn=2 -d . hello-world.sh |

### Build process (DevCloud)
1. Build the program using the following `cmake` commands.
```
$ mkdir build && cd build
$ cmake ..
$ make
```

## Running the `PointPillars` Sample Program (DevCloud)
After a successful build, the sample program can be run as follows:
```
./example.exe
```
The input data for the sample program is the example.pcd located in the /data folder. It contains an artificial point cloud from a simulated LIDAR sensor of the CARLA simulator [3]. The corresponding scene looks as follows:
![SampleScene](data/image.png)
The example.pcd file is a point cloud in ASCII using the PCL-format, which renders to (using the pcl_viewer tool):
![SamplePointCloud](data/pointcloud.png)
It is worth noting that there are three cars in this scene, one of which is very far away and thus not yet properly covered in the LIDAR scan. Then, there is a black car inside the intersection, which is also well visible in the LIDAR data, and a hidden car, right behind the black one. Hence, `PointPillars` should detect at least one car.

Successful execution of the sample program results in at least one detected object and an output similar to:
```
Starting PointPillars
   PreProcessing - 20ms
   AnchorMask - 10ms
   PFE Inference - 91ms
   Scattering - 50ms
   RPN Inference - 107ms
   Postprocessing - 13ms
Done
Execution time: 296ms

1 cars detected
Car: Probability = 0.622569 Position = (24.8561, 12.5615, -0.00771689) Length = 2.42855 Width = 3.61396
```

## Execution Options for the Sample Program
The sample program provides a few command-line options, which can be accessed using the 'help' option
```
./example.exe --help
```

Furthermore, it is possible to specify the execution device. For using multi-threaded CPU execution, please use:
```
./example.exe --cpu
```
For single-threaded execution on the host system, please use:
```
./example.exe --host
```
And to use an Intel® DG1 or integrated graphics, please use:
```
./example.exe --gpu
```
These options can also be used in combination, e.g.:
```
./example.exe --cpu --gpu --host

## Known Limitations
- This sample code only works using models trained with the Sigmoid function, not with Softmax or Crossentropy
- If other models than the recommended models are used, it has to be ensured that the maximum number of classifications is at most 20


---

## References
[1] [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

[2] [Autoware package for Point Pillars](https://github.com/Autoware-AI/core_perception/tree/master/lidar_point_pillars)

[3] [Open-source simulator for autonomous driving research](http://carla.org/)

## Notes
OpenVINO is a trademark of Intel Corporation or its subsidiaries
