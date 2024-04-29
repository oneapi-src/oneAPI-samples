# Intel&reg; Rendering Toolkit Learning Path
  
## Requirements
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 22.04
| Software                          | Intel&reg; Rendering Toolkit (Render Kit), Jupyter Notebooks, Intel&reg; Developer Cloud (IDC)

## Description
This repo contains Jupyter Notebook Trainings for Render Kit that **has been designed to be used on the Intel Developer Cloud** for hands-on workshops.

At the end of this course, you will be able to:

- Create high-fidelity photorealistic images using the Intel OSPRay renderer.
- Use the Embree API to execute ray-surface intersection tests required for performant ray-tracing applications.
- Use the Open VKL API to execute ray-volumetric hit queries required for performant rendering of volumetric objects.
- Use Intel Open Image Denoise to reducing the amount of necessary samples per pixel in ray tracing-based rendering applications by filtering out noise inherent to stochastic ray tracing methods.
- Use Intel OSPRay Studio to create rendered pictures by seting up the relevant parameters in its interactive interface.

## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Content Details

### Pre-requisites
- C++ Programming

### Training Modules

| Modules | Description
|---|---|
|[Render Kit Introduction](1_RenderKit_Intro/RenderKit_Intro.ipynb)| + Introduction and Motivation for the Intel Rendering Toolkit components.
|[Intel® OSPRay tutorial with CPU & GPU](2.1_OSPRay_Intro_CPU_GPU/OSPRay_tutorial_CPU_GPU.ipynb)| + An introduction to OSPRAY, a high-performance ray-tracing renderer for scientific visualization and high-fidelity photorealistic rendering.
|[Intel® OSPRay tutorial with denoise](2.1_OSPRay_Intro_CPU_GPU/OSPRay_tutorial_denoise.ipynb)| + A basic tutorial on how to implement Open Image Denoise into OSPRAY so the output images are denoised.
|[Intel® Embree minimal with CPU](3.1_Embree_Intro_CPU_GPU/Embree_minimal_CPU.ipynb)| + Getting started with Embree. A demonstration of how to initialize a device and scene, and how to intersect rays with the scene targeting the CPU.
|[Intel® Embree minimal with GPU](3.1_Embree_Intro_CPU_GPU/Embree_minimal_GPU.ipynb)| + Getting started with Embree. A demonstration of how to initialize a device and scene, and how to intersect rays with the scene targeting the GPU.
|[Intel® Embree minimal with shadow](3.2_Embree_Shadow/Embree_minimal_shadow.ipynb)| + An introduction on how to check if an area is shadow with Embree.
|[Intel® Open VKL minimal with CPU](4.1_OpenVKL_Intro_CPU_GPU/OpenVKL_minimal_CPU.ipynb)| + A step-by-step demonstration of how Open VKL can be introduced in a simple code targeting the CPU.
|[Intel® Open VKL minimal with GPU](4.1_OpenVKL_Intro_CPU_GPU/OpenVKL_minimal_gPU.ipynb)| + A step-by-step demonstration of how Open VKL can be introduced in a simple code targeting the GPU.
|[Intel® Open VKL Tutorial with CPU & GPU](4.2_OpenVKL_Tutorial/OpenVKL_tutorial.ipynb)| + This module creates a simple procedural regular structured volume and uses the various API version iterate using scalar, vector, and stream methods.  This module shows how to target both, CPUs and GPUs.
|[Intel® Open Image Denoise with CPU & GPU](5_OIDN_Intro_CPU_GPU/OIDN_Intro_CPU_GPU.ipynb)| + Code walkthrough of the Intel Open Image Denoise library and how to target CPUs and GPUs.
|[Intel® OSPRay Studio tutorial](6_OSPRay_Studio/OSPRay_Studio.ipynb)| + Basic demonstration on how to use Intel® OSPRay Studio with its interactive interface.


### Content Structure

Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in Jupyter Lab to view the training contant, edit code and compile/run. Along with the Notebook file, there is a `lab` and a `src` folder with SYCL source code for samples used in the Notebook. The module folder also has `build_*.sh` and `run_*.sh` files which can be used in shell terminal to compile and run each sample code.

### Access using Intel Developer Cloud

The Jupyter notebooks are tested and can be run on Intel Developer Cloud without any installation necessary, below are the steps to access these Jupyter notebooks on Intel Developer Cloud:
1. Register on [Intel Developer Cloud (IDC).](https://console.cloud.intel.com/)
2. Create an account and/or log in.
3. Once in IDC dashboard, go to Training in the left panel and then click the Launch JupyterLab botton in the upper right.
4. Open Terminal in Jupyter Lab and git clone the repo and access the Notebooks.

Note that the oneAPI Base Toolkit and the Intel Rendering Toolkit are already pre-installed in IDC.

### Workaround for 2-Tile Intel&reg; Data Center GPU Max Series (PVC)

Intel OSPRay, Intel Embree, and Intel OpenVKL based programs need to use the environment variable ZE_FLAT_DEVICE_HIERARCHY and set it to COMPOSITE.

Use the following command in your shell or Jupyter Notebook cell before running programs:

`export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`
