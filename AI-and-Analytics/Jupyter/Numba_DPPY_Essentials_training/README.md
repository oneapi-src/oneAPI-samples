# Numba Data parallel python training Jupyter notebooks

The purpose of this repo is to be the central aggregation, curation, and
distribution point for Juypter notebooks that are developed in support of
Numba Data parallel python training programs.

The Jupyter notebooks are tested and can be run on the Intel Devcloud. Below
are the steps to access these Jupyter notebooks on the Intel Devcloud:

1. Register with the Intel Devcloud at
   https://intelsoftwaresites.secure.force.com/devcloud/oneapi

2. SSH into the Intel Devcloud "terminal"

3. Type the following command to download the Numba Data parallel Python series of
   Jupyter notebooks into your devcloud account
   `/data/oneapi_workshop/get_jupyter_notebooks.sh`

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Organization of the Jupyter Notebook Directories

Notebook Name: Owner
* Descriptions

[DPPY Intro](01_DPPY_Intro): Praveen.K.Kundurthy@intel.com
* Introduction and Motivation for Data parallel python: These initial hands-on exercises introduce you to concepts of Data Parallel Python. In addition, it familiarizes you how to execute on multiple devices using Data Parallel Python (DPPY), utilize Numba and Numba-DPPY to write paralle code on GPU 
* Intro to Numba-Dppy
* @njit Decorator:Explicit and Implicit offload
* @dppy.kernel decorator
* _Lab Excercise_: Matrix multiplication using numba_dppy

[DPCTL Intro](02_dpctl_Intro): Praveen.K.Kundurthy@intel.com
* __Classes__ - device, device_selector, queue using dpctl
* USM and memory management using __dpctl__

[DPPY Pairwise Distance Algorithm](03_DPPY_Pairwise_Distance): Praveen.K.Kundurthy@intel.com
* Pairwise distance algorithm targeting CPU and GPU using __Numba__ Jit decorator
* Pairwise distance algorithm targeting GPU using __Kernel__ decorator
* Pairwise distance algorithm targeting GPU using __Numpy__ approach

[DPPY Black Scholes Algorithm](04_DPPY_Black_Sholes): Praveen.K.Kundurthy@intel.com
* Black Scholes algorithm targeting CPU and GPU using __Numba_ Jit decorator
* Black Scholes algorithm targeting GPU using __Kernel__ decorator
* Black Scholes algorithm targeting GPU using __Numpy__ approach

[DPPY K-Means Algorithm](05_DPPY_Kmeans): Praveen.K.Kundurthy@intel.com
* K-Means algorithm targeting CPU and GPU using __Numba__ Jit decorator
* K-Means algorithm targeting GPU using __Kernel__ decorator
* K-Means algorithm targeting GPU using __Numpy__ and __Atomics__

[DPPY Gpairs Algorithm](05_DPPY_GPAIRS): Praveen.K.Kundurthy@intel.com
* Gpairs algorithm targeting CPU and GPU using __Numba__ Jit decorator
* Gpairs algorithm targeting GPU using __Kernel__ decorator

__Note__: Please take care to secure the connection while using functions like __Pickle__ and establish an appropriate trust verification mechanism.


