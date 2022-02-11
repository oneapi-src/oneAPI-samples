# Intel&reg; oneAPI AI Analytics Toolkit Container Sample

Containers allow you to set up and configure environments for
building, running, and profiling oneAPI AI applications and distribute
them using images.


## Purpose

This sample code shows how to get started with Intel® oneAPI AI
Analytics Toolkit container. It provides a Bash script to help users
configure their aikit container environment. Developers can quickly
build and train a neural network using this Docker* environment.

For more information on the one API AIKit container, see [oneAPI AIKit
Container Repository](https://hub.docker.com/r/intel/oneapi-aikit).


| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | Intel® Xeon® Scalable processor family or newer
| Software                          | Intel® AI Analytics Toolkit
| What you will learn               | How to start using the Intel® oneapi-aikit container
| Time to complete                  | 10 minutes


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
complete instructions for using the oneAPI AI Analytics containers see
the [Getting Started Guide]
(https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top/using-containers.html.)

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

## Manage Docker* Images

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


## Next Steps

Explore the [Get Started
Guide](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html).


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
