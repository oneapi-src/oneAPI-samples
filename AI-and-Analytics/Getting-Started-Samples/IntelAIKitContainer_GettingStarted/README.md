﻿# Intel&reg; AI Analytics Toolkit (AI Kit) Container Sample

Containers allow you to set up and configure environments for
building, running, and profiling AI applications and distribute
them using images. You can also use Kubernetes* to automate the
deployment and management of containers in the cloud.

This get started sample shows the easiest way to start using any of
the [Intel® AI Analytics
Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
components without the hassle of installing the toolkit, configuring
networking and file sharing.


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | Intel® Xeon® Scalable processor family or newer
| Software                          | Intel® AI Analytics Toolkit
| What you will learn               | How to start using the Intel® oneapi-aikit container
| Time to complete                  | 10 minutes

## Purpose

This sample provides a Bash script to help users configure their Intel&reg; AI Analytics Toolkit 
container environment. Developers can
quickly build and train deep learning models using this Docker*
environment.

For more information on the one API AIKit container, see [AI Kit
Container Repository](https://hub.docker.com/r/intel/oneapi-aikit).


## Key Implementation Details

The Bash script provided in this sample performs the following
configuration steps:

- Mounts the `/home` folder from host machine into the Docker
  container. You can share files between the host machine and the
  Docker container via the `/home` folder.

- Applies proxy settings from the host machine into the Docker
  container.
   
- Uses the same IP addresses between the host machine and the Docker
  container.

- Forwards ports 8888, 6006, 6543, and 12345 from the host machine to
  the Docker container for some popular network services, such as
  Jupyter notebook and TensorBoard.
        

## Run the Sample

This sample uses a configuration script to automatically configure the
environment. This provides fast and less error prone setup. For
complete instructions for using the AI Kit containers see
the [Getting Started Guide](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top/using-containers.html.)

To run the configuration script on Linux*, type the following command
in the terminal with [Docker](https://docs.docker.com/engine/install/)
installed:


1. Navigate to the directory with the IntelAIKitContainer sample and pull the oneapi-aikit docker image:

    ```
    docker pull intel/oneapi-aikit
    ```
    > Please apply the below command and login again if a permisson denied error occurs.
    ```
    sudo usermod -aG docker $USER
    ```
    
2. Use the `run_oneapi_docker.sh` Bash script to run the Docker image:

   ```bash
   ./run_oneapi_docker.sh intel/oneapi-aikit
   ```

   The script opens a Bash shell inside the Docker container.
   > Note : Users could install additional packages by adding them into requirements.txt.   
   > Please copy the modified requirements.txt into /tmp folder, so the bash script will install those packages for you.
    
   To create a new Bash session in the running container from outside
   the Docker container, use the following:
		
   ```bash
   docker exec -it aikit_container /bin/bash
   ```
   
3. In the Bash shell inside the Docker container, activate the oneAPI
   environment:
    
   ```bash
   source activate tensorflow
   ```
   
   or
   
   ```bash
   source activate pytorch
   ```
   
Now you can start using Intel® Optimization for TensorFlow* or Intel
Optimization for PyTorch inside the Docker container.
   
To verify the activated environment, navigate to the directory with
the IntelAIKitContainer sample and run the `version_check.py` script:
   
```bash
python version_check.py
```

## Example of Output

Output from TensorFlow Environment
```
TensorFlow version:  2.6.0
MKL enabled : True
```

Output from PyTorch Environment
```
PyTorch Version:  1.8.0a0+37c1f4a
mkldnn : True,  mkl : True, openmp : True
```



## Next Steps

Explore the [Get Started
Guide](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html).
to find out how you can achieve performance gains for popular
deep-learning and machine-learning frameworks through Intel
optimizations.

### Manage Docker* Images

You can install additional packages, upload the workloads via the
`/tmp` folder, and then commit your changes into a new Docker image,
for example, `intel/oneapi-aikit-v1`:


```bash
docker commit -a "intel" -m "test" DOCKER_ID  intel/oneapi-aikit-v1
```

**NOTE:** Replace `DOCKER_ID` with the ID of your container. Use
`docker ps` to get the DOCKER_ID of your Docker container.

You can then use the new image name to start Docker:

```bash
./run_oneapi_docker.sh intel/oneapi-aikit-v1
```

To save the Docker image as a tar file:

```bash
docker save -o oneapi-aikit-v1.tar intel/oneapi-aikit-v1
```

To load the tar file on other machines:

```bash
docker load -i oneapi-aikit-v1.tar
```
## Troubleshooting

### Docker Proxy

#### Ubuntu
For docker proxy related problem, you could follow below instructions to setup proxy for your docker client.

1. Create a new directory for our Docker service configurations
```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
```
2. Create a file called proxy.conf in our configuration directory.
```bash
sudo vi /etc/systemd/system/docker.service.d/proxy.conf
```
3. Add the following contents, changing the values to match your environment.
```bash
[Service]
Environment="HTTP_PROXY=http://proxy-hostname:911/"
Environment="HTTPS_PROXY="http://proxy-hostname:911/
Environment="NO_PROXY="10.0.0.0/8,192.168.0.0/16,localhost,127.0.0.0/8,134.134.0.0/16"
```
4. Save your changes and exit the text editor.
5. Reload the daemon configuration
```bash
sudo systemctl daemon-reload
```
6. Restart Docker to apply our changes
```bash
sudo systemctl restart docker.service
```
## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
