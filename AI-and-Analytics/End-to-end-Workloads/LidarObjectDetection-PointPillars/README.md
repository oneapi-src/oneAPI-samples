# `Lidar Object Detection using PointPillars` Sample
The `Lidar Object Detection using PointPillars` sample demonstrates how to perform 3D object detection and classification using input data (point cloud) from a LIDAR sensor. 

>**Note**: This sample implementation is based on the [*PointPillars: Fast Encoders for Object Detection from Point Clouds*](https://arxiv.org/abs/1812.05784) paper and the implementation in [Point Pillars for 3D Object Detection](https://github.com/Autoware-AI/core_perception/tree/master/lidar_point_pillars) GitHub repository. 

| Area                     | Description
|:---                      |:---
| What you will learn      | How to combine Intel® Distribution of OpenVINO™ toolkit and Intel® oneAPI Base Toolkit (Base Kit) to offload the computation of a complex workload to CPU or GPU
| Time to complete         | 30 minutes

## Purpose

This sample is an Artificial Intelligence (AI) algorithm that uses LIDAR point clouds to detect and classify 3D objects in the sensor environment. For this purpose, the algorithm consists of the steps shown in the image and described below.

![Overview](data/point_pillars_overview.png)

1. The LiDAR input point cloud is pre-processed with the help of kernels implemented using SYCL.
2. Using SYCL, the sample generates an anchor grid. The anchors in the grid are used in object detection to refine detected boxes by the Region Proposal Network (RPN).
3. A Pillar Feature Extraction (PFE) Convolutional Neural Network (CNN) uses the pre-processed data to create a 2D image-like representation of the sensor environment. For the inference, this sample uses the Intel® Distribution of OpenVINO™ toolkit. The output of this CNN is a list of dense tensors, or learned pillar features.
4. Using SYCL, the sample performs a scatter operation to convert these dense tensors into a pseudo-image.
5. A second CNN, Region Proposal Network (RPN), consumes the pseudo-image. The inference is performed with the help of the Intel® Distribution of OpenVINO™ toolkit. The output is an unfiltered list of possible object detections, their position, dimensions, and classifications.
6. Using SYCL kernels, the output data (object list) is post-processed with the help of the anchors created earlier in Step 2. The anchors are used to decode the object position, dimension, and class. A Non-Maximum-Suppression (NMS) is used to filter out redundant/clutter objects.
7. Using SYCL kernels, the objects are sorted according to likelihood and provided as output.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 (or newer) <br>  Intel® Iris® X<sup>e</sup> graphics
| Software                          | Intel® oneAPI Base Toolkit (2021.2 or newer) <br> Intel® Distribution of OpenVINO™ toolkit (2022.1 or newer)

Additionally, you must install the following:

- **Boost** including the `boost::program_options` and `boost::filesystem` libraries. For Ubuntu*, you can install the `libboost-all-dev` package.
- (Optional) If you plan to run the sample on Intel GPUs, you might need to upgrade the relevant drivers. See [https://github.com/intel/compute-runtime/releases/](https://github.com/intel/compute-runtime/releases/) for current information.
- (Optional) **Point Cloud Viewer** tool. For Ubuntu*, you can install the `pcl-tools` package.

## Key Implementation Details

This sample demonstrates a real-world, end-to-end example that will give you insights many SYCL development concepts. By running the sample and reading the code, you will learn about the following concepts.
 - Transferring data from a SYCL device/kernel to an OpenVINO™-based inference task and back.
 - Implementing a device manager that allows choosing the target hardware for execution (for example, CPU, GPU, or an accelerator) at runtime in a user transparent manner. The sample program allows you to choose the device using a command-line argument without requiring a time-consuming re-compilation. Details about this functionality provide below.
 - Implementing oneAPI-based function kernels to execute on the host system, on a multi-threaded CPU, or a GPU.
 - Using oneAPI to implement standard algorithms, like *Non-Maximum-Suppression*, for AI-based object detection


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

> **Note**: You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

## Build the `Lidar Object Detection using PointPillars` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For Base Kit system wide installations: `. /opt/intel/oneapi/setvars.sh` 
> - For Base Kit private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
> - For OpenVINO™: `source <INSTALL_DIR>/setupvars.sh`. For example, `source /opt/intel/openvino_2021/bin/setupvars.sh`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) and [Install and Configure Intel® Distribution of OpenVINO™ Toolkit for Linux](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html).

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux

1. Build the OpenCL dependencies
```
cd /tmp
git clone --recursive https://github.com/KhronosGroup/OpenCL-SDK.git
cd OpenCL-SDK && mkdir build && cd build
cmake .. -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DOPENCL_SDK_BUILD_OPENGL_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$HOME/local
make install
```
2. Change to the sample directory.
3. Build the program.
   ```
   mkdir build && cd build
   cmake .. -DCMAKE_PREFIX_PATH=$HOME/local
   make
   ```
> **Note***: cmake will download the **ONNX models** required for the two inference steps executed with the Intel® Distribution of OpenVINO™ toolkit.

If an error occurs, you can get more details by running `make` with the
`VERBOSE=1` argument:
 ```
 make VERBOSE=1
 ```

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `Lidar Object Detection using PointPillars` Program

### Configurable Parameters

You can specify command-line options to target spefic devices.
```
program --<option>
```
The following table lists and describes the available options.

|Option       | Description
|:---         |:---
| `--help`    | Get help on using the program.
| `--cpu`     | Specify CPU as the execution device.
| `--host`    | Specify single-threaded execution.
| `--gpu`     | Specify a Intel® DG1 or integrated graphics.

>**Note**: You can combine the options. For example, `./example.exe --cpu --gpu --host`.

By default, the program uses `--host` as the device for SYCL kernels and `--cpu` (single-threaded) for Intel® Distribution of OpenVINO™ toolkit inferencing. The execution device and the inferencing device are displayed in the output, along with the elapsed time for each step described earlier.

1. Change to the output directory.
2. Run the sample program.
   ```
   ./example.exe --cpu
   ```
3. Clean the program. (Optional)
   ```
   make clean
   ```

## Example Output

The input data for the sample program is the `example.pcd` file located in the **/data** folder. It contains an artificial point cloud from a simulated LIDAR sensor from [CARLA Open-source simulator for autonomous driving research](http://carla.org/). 

This image shows the corresponding scene.

![SampleScene](data/image.png)

There are three cars in this scene, one of which is far away and is not yet properly covered in the LIDAR scan. There is a black car in the intersection, which is also visible in the LIDAR data, and a car behind the black one, which is hidden. The sample code should detect at least one car.

The `example.pcd` file is a point cloud in ASCII using the PCL-format, which renders something similar to the following in the Point Cloud Viewer (pcl_viewer) tool.

![SamplePointCloud](data/pointcloud.png)

Using the data provided in the sample, the program should detect at least one object.

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

## Build and Run the `Lidar Object Detection using PointPillars` Sample in Intel® DevCloud (Optional)
<This is the long version. Use ONLY the short version OR the long version NOT both.>

When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information on using Intel® DevCloud, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

### Build and Run Samples in Batch Mode

This sample runs in batch mode, so you must have a script for batch processing. Once you have a script set up, continue with the next section, Request a Compute Node.

You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`. 

If you choose to use scripts, jobs terminate with writing files to the disk:
- `<script_name>.sh.eXXXX`, which is the job stderr
- `<script_name>.sh.oXXXX`, which is the job stdout

Here XXXX is the job ID, which gets printed to the screen after each qsub command. 

You can inspect output of the sample.
```
cat run.sh.oXXXX
```
You must provide the node when submitting a job to run your sample in batch mode using the qsub command. If you need more information on creating qsub scripts, see the [*Run a Hello World! Sample*](https://devcloud.intel.com/oneapi/get_started/baseToolkitSamples/) section of the Intel® DevCloud for oneAPI *Get Started* article.

### Request a Compute Node

In order to run on Intel DevCloud, you need to request a compute node using node properties such as: `gpu`, `xeon`, `fpga_compile`, `fpga_runtime` and others. For more information about the node properties, execute the `pbsnodes` command.

Change the command to fit the node you are using. Compatible nodes for this sample are listed below.

| Node     | Command Options
|:---      |:---
| GPU      | qsub -l nodes=1:gpu:ppn=2 -d . \<name\>.sh
| CPU      | qsub -l nodes=1:xeon:ppn=2 -d . \<name\>.sh

where:
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node.
   - `-d .` makes the current folder as the working directory for the task.

>**Note**: For more information on how to specify compute nodes read, [Launch and manage jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the Intel® DevCloud for oneAPI Documentation.

#### Build and Run on Intel® DevCloud

1. Open a terminal on a Linux* system.
2. Log in to the Intel® DevCloud.
   ```
   ssh devcloud
   ```
3. Download the samples from GitHub. (If you have not already done so.)
   ```
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```
4. Create a batch mode script. (If you have not already done so.)
5. Configure the sample for a cpu node.
6. Change to the sample directory.
7. Perform build steps you would on Linux. (Including optionally cleaning the project.)
8. Run the sample.
9. Disconnect from the Intel® DevCloud.
	```
	exit
	```

## Known Limitations

- This sample code works using models trained with the Sigmoid function only. It will not work using models trained with Softmax or Crossentropy.
- If you use models other than the recommended models, you must ensure that the maximum number of classifications does not exceed 20.


## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
