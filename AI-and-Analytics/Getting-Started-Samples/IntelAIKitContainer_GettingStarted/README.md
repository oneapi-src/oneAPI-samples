# `Intel oneAPI AI Kit Container` Sample
Containers allow you to set up and configure environments for building, running and profiling oneAPI AI applications and distribute them using images. 

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | Intel® Xeon® Scalable processor family or newer
| Software                          | Intel® AI Analytics Toolkit
| What you will learn               | How to get started to use Intel® oneapi-aikit container
| Time to complete                  | 10 minutes

## Purpose
This sample code shows how to get started with Intel® oneAPI AI Kit Container*. It provides a bashscript to help users configure their aikit container environment. Developers can quickly build and train a neural network using a this docker environment. 

For more information on aikit container, please check the [oneAPI AIKit Container Repository](https://hub.docker.com/r/intel/oneapi-aikit).

## Key implementation details
1. mount /home folder from host machine into docker instance. Users could share files between the host machine and the docker instance via /home folder.
2. apply proxy settings from the host machine into docker instance
3. use same IP addresses between the host machine and docker instance.
4. forward port 8888, 6006, 6543, and 12345 from host machine to docker instance for some popular network services such as jupyter notebook and tensorboard.
        
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Running the Sample

To run the program on Linux*, type the following command in the terminal with [Docker](https://docs.docker.com/engine/install/) installed:

1. Navigate to the directory with the IntelAIKitContainer sample and pull the oneapi-aikit docker image:

    ```
    docker pull intel/oneapi-aikit
    ```
2. Run the docker image via a bash script:  
        

    ```
    ./run_oneapi_docker.sh intel/oneapi-aikit
    ```    

    After the command execution, users will get into a bash shell inside docker instance.  
    
    > Users could use below command outside of docker instance to login into the running instance.   
    ```
    docker exec -it aikit_container /bin/bash
    ```
3. Activate oneAPI environment and start to use Intel® Optimization for TensorFlow* or  Intel Optimization for PyTorch:   
    
    First, get the version_check.py either via your local copy or get it from github directly following below command.
    ```
    wget https://raw.githubusercontent.com/intel-ai-tce/oneAPI-samples/oneapi_docker/AI-and-Analytics/Getting-Started-Samples/IntelAIKitContainer_GettingStarted/version_check.py
    ```
    
    
    Intel® Optimization for TensorFlow*   
        
    ```
    source /opt/intel/oneapi/setvars.sh
    source activate tensorflow
    python version_check.py
    ```   
        
    Intel Optimization for PyTorch   
        
    ```
    source /opt/intel/oneapi/setvars.sh
    source activate pytorch
    python version_check.py
    ```
4. users could install additional packages and upload the workloads via /tmp folder and then commit your works into a new docker image "intel/oneapi-aikit-v1":    

    > NOTE : Please replace DOCKER_ID with the related ID of your instance, users could use "docker ps" to get the DOCKER_ID of your docker instance.

    ```
    docker commit -a "intel" -m "test" DOCKER_ID  intel/oneapi-aikit-v1
    ```    
    > NOTE : please use the new image name to start the docker like "./run_oneapi_docker.sh intel/oneapi-aikit-v1" for your committed works.   

5. users could save the docker image as a image tar file    

    ```
    docker save -o oneapi-aikit-v1.tar intel/oneapi-aikit-v1
    ```
6. users could also load the tar file on other machines  

    ```
    docker load -i oneapi-aikit-v1.tar
    ```


