# `Intel® AI Tools Container Getting Started` Sample
>**Note**: This sample is relevant only for AI Tools installed via Docker. If you have installed AI tools using PIP or Conda, this sample may not be relevant for you.

The `Intel® AI Tools Container Getting Started` sample demonstrates how to use AI Tools containers.  

| Area                  | Description
|:---                   |:---
| What you will learn   | How to start using the Intel® AI Tools container
| Time to complete      | 10 minutes
| Category              | Tutorial

For more information on the AI Tools container, see [Intel Deep Learning](https://hub.docker.com/r/intel/deep-learning), [Intel Machine Learning](https://hub.docker.com/r/intel/classical-ml), [Intel Data Analytics](https://hub.docker.com/r/intel/data-analytics), and [Intel Inference Optimization](https://hub.docker.com/r/intel/inference-optimization) Docker Hub pages.

## Purpose

This sample provides a Bash script to help you configure an AI Tools container environment. You can build and train deep learning models using this Docker* environment.

Containers allow you to set up and configure environments for building, running, and profiling AI applications and distribute them using images. You can also use Kubernetes* to automate the deployment and management of containers in the cloud.

Read the [Get Started with the Intel® AI Tools for Linux*](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) to find out how you can achieve performance gains for popular deep-learning and machine-learning frameworks through Intel optimizations.

This sample shows an easy way to start using any of the [Intel® AI Tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html) components without the hassle of installing the toolkit, configuring networking and file sharing.

## Prerequisites

| Optimized for      | Description
|:---                |:---
| OS                 | Ubuntu* 20.04 (or newer)
| Hardware           | Intel® Xeon® Scalable processor family
| Software           | Intel® AI Tools Container

## Key Implementation Details

The Bash script provided in this sample performs the following
configuration steps:

- Mounts the `/home` folder from host machine into the Docker container. You can share files between the host machine and the Docker container through the `/home` folder.

- Applies proxy settings from the host machine into the Docker container.

- Uses the same IP addresses between the host machine and the Docker container.

- Forwards ports 8888, 6006, 6543, and 12345 from the host machine to the Docker container for some popular network services, such as Jupyter* Notebook and TensorFlow* TensorBoard.
  
- Enable VTune Profiling
 
## Run the `Intel® AI Tools Deep Learning Container Getting Started` Sample

This sample uses a configuration script to automatically configure the environment. This provides fast and less error prone setup. For complete instructions for using the AI Tools container, see the [Getting Started Guide](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top/using-containers.html).

### On Linux*

You must have [Docker](https://docs.docker.com/engine/install/)
installed.

1. Open a terminal.
2. Change to the sample folder, and pull the AI Tools Deep Learning Docker image by following [AI Tools Selector page](www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html).
   ex: 
   ```
   docker pull intel/deep-learning:2024.0-py3.10
   ```
   >**Note**: If a permission denied error occurs, run the following command.
   >```
   >sudo usermod -aG docker $USER
   >```

4. Run the Docker images using the `run_oneapi_docker.sh` Bash script.
   ```
   ./run_oneapi_docker.sh intel/deep-learning:2024.0-py3.10
   ```
   The script opens a Bash shell inside the Docker container and name the docker instance as "aitools_container" by default.
   > **Note**: Install additional packages by adding them into requirements.txt file in the sample. Copy the modified requirements.txt into /tmp folder, so the bash script will install those packages for you.

   To create a Bash session in the running container from outside the Docker container, enter a command similar to the following.
   ```
   docker exec -it aitools_container /bin/bash
   ```
5. In the Bash shell inside the Docker container, activate the specialized environment.
   ```
   source activate tensorflow
   ```
   or
   ```
   source activate torch
   ```
You can start using Intel® Optimization for TensorFlow* or Intel® Optimization for PyTorch* inside the Docker container.

>**Note**: You can verify the activated environment. Change to the directory with the IntelAIKitContainer sample and run the `version_check.py` script.
>```
>python version_check.py
>```

### Manage Docker* Images

You can install additional packages, upload the workloads via the `/tmp` folder, and then commit your changes into a new Docker image, for example, `intel/deep-learning-v1`.
```
docker commit -a "intel" -m "test" DOCKER_ID  intel/deep-learning-v1
```
>**Note**: Replace `DOCKER_ID` with the ID of your container. Use `docker ps` to get the DOCKER_ID of your Docker container.

You can use the new image name to start Docker.
```
./run_oneapi_docker.sh intel/deep-learning-v1
```

You can save the Docker image as a tar file.
```
docker save -o oneapi-aikit-v1.tar intel/deep-learning-v1
```

You can load the tar file on other machines.
```
docker load -i deep-learning-v1.tar
```

### Docker Proxy

For Docker proxy related problem, you could follow below instructions to configure proxy settings for your Docker client.

1. Create a directory for the Docker service configurations.
   ```
   sudo mkdir -p /etc/systemd/system/docker.service.d
   ```
2. Create a file called `proxy.conf` in our configuration directory.
   ```
   sudo vi /etc/systemd/system/docker.service.d/proxy.conf
   ```
3. Add the contents similar to the following to the `.conf` file. Change the values to match your environment.
   ```
   [Service]
   Environment="HTTP_PROXY=http://proxy-hostname:911/"
   Environment="HTTPS_PROXY="http://proxy-hostname:911/
   Environment="NO_PROXY="10.0.0.0/8,192.168.0.0/16,localhost,127.0.0.0/8,134.134.0.0/16"
   ```
4. Save your changes and exit the text editor.
5. Reload the daemon configuration.
   ```
   sudo systemctl daemon-reload
   ```
6. Restart Docker to apply our changes.
   ```
   sudo systemctl restart docker.service
   ```

## Example Output

### Output from TensorFlow* Environment

```
TensorFlow version:  2.6.0
MKL enabled : True
```

### Output from PyTorch* Environment

```
PyTorch Version:  1.8.0a0+37c1f4a
mkldnn : True,  mkl : True, openmp : True
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
