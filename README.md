# oneAPI Samples

The oneAPI-samples repository contains samples for the [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html).

The contents of the default branch in this repository are meant to be used with the most recent released version of the Intel® oneAPI Toolkits.

## Find oneAPI Samples

You can find samples by browsing the *[oneAPI Samples Catalog](https://oneapi-src.github.io/oneAPI-samples/)*. Using the catalog you can search on the sample titles or descriptions.

You can refine your browsing or searching through filtering on the following:

- Expertise (Getting Started, Tutorial, etc.)
- Programming language (C++, Python, or Fortran)
- Target device (CPU, GPU, and FPGA)

## Get the oneAPI Samples

Clone the repository by entering the following command:

`git clone https://github.com/oneapi-src/oneAPI-samples.git`

Alternatively, you can download a zip file containing the primary branch in repository.

1. Click the **Code** button.
2. Select **Download ZIP** from the menu options.
3. After downloading the file, unzip the repository contents.

### Get Earlier Versions of the oneAPI Samples

If you need samples for an earlier version of any of the Intel® oneAPI Toolkits, then use a [tagged version](https://github.com/oneapi-src/oneAPI-samples/tags) of the repository that corresponds with the toolkit version.

Clone an earlier version of the repository using Git by entering a command similar to the following:

`git clone -b <tag> https://github.com/oneapi-src/oneAPI-samples.git`

where `<tag>` is the GitHub tag corresponding to the toolkit version number, like **2025.2.0**.

Alternatively, you can download a zip file containing a specific tagged version of the repository.

1. Select the appropriate tag.
2. Click the **Code** button.
3. Select **Download ZIP** from the menu options.
4. After downloading the file, unzip the repository contents.

## Getting Started with oneAPI Samples

The best oneAPI sample to start with depends on what you are trying to learn or types of problems you are trying to solve.

| If you want to learn about...                                                        | Start with...
|:---                                                                                  |:---
| the basics of writing, compiling, and building programs for CPUs and GPUs            |[Simple Add](https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming/C++SYCL/DenseLinearAlgebra/simple-add) or [Vector Add](https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming/C++SYCL/DenseLinearAlgebra/vector-add) samples <br> (You can use these samples as starter projects by removing unwanted elements and adding your code and build requirements.)
| the basics of using artificial intelligence                                          | [Getting Started Samples for AI Tools](https://github.com/oneapi-src/oneAPI-samples/tree/main/AI-and-Analytics/Getting-Started-Samples)

>**Note**: The README.md included with each sample provides build instructions for all supported operating systems. For samples that run in Jupyter Notebooks, you may need to install or configure additional frameworks or package managers if you do not already have them on your system.

### Using Integrated Development Environments (IDE)

If you prefer to use an Integrated Development Environment (IDE) with these samples, you can download [Visual Studio Code](https://code.visualstudio.com/download) for use on Windows or Linux.

## Repository Structure

The oneAPI-sample repository is organized by high-level categories.

- [AI-and-Analytics](https://github.com/oneapi-src/oneAPI-samples/tree/main/AI-and-Analytics)
  - [End-to-End-Workloads](https://github.com/oneapi-src/oneAPI-samples/tree/main/AI-and-Analytics/End-to-end-Workloads)
  - [Features-and-Functionality](https://github.com/oneapi-src/oneAPI-samples/tree/main/AI-and-Analytics/Features-and-Functionality)
  - [Getting-Started-Samples](https://github.com/oneapi-src/oneAPI-samples/tree/main/AI-and-Analytics/Getting-Started-Samples)
- [DirectProgramming](https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming)
  - [C++](https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming/C++)
  - [C++SYCL](https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming/C++SYCL)
  - [Fortran](https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming/Fortran)
- [Libraries](https://github.com/oneapi-src/oneAPI-samples/tree/main/Libraries)
- [Publications/GPU-Opt-Guide](https://github.com/oneapi-src/oneAPI-samples/tree/main/Publications/GPU-Opt-Guide)
- [Tools](https://github.com/oneapi-src/oneAPI-samples/tree/main/Tools/)

## Platform Validation

### Ubuntu 22.04.3
Intel(R) Xeon(R) Platinum 8468V \
Intel(R) Data Center GPU Max 1100 \
OpenCL Driver: Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8468V OpenCL 3.0 (Build 0) [2025.20.5.0.15_220340] \
Level Zero Driver: Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1100 12.60.7 [1.3.27642] \
oneAPI package version: \
&dash; Intel oneAPI HPC Toolkit Build Version: 2025.2.0.463

## Known Issues and Limitations

### Windows

- If you are using Microsoft Visual Studio* 2019, you must use Microsoft Visual Studio 2019 version 16.4.0 or newer.
- If you encounter `Error MSB6003 The specified task executable ... could not be run...` when building a sample program, it might be due to the length of the directory path. Move the `build` directory to a location with a shorter path. Build the sample in the new location.

## Additional Resources for Code Samples
A curated list of samples from oneAPI based projects, libraries, and tools. In addition, the most exciting samples from other AI projects that are not necessarily based on oneAPI are also listed here to provide you with the latest and valuable resources for augmenting your productivity.
-	[OpenVINO™ notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks): A collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO™ Toolkit, an open-source AI toolkit that makes it easier to write once, deploy anywhere. The notebooks introduce OpenVINO basics and teach developers how to leverage the API for optimized deep learning inference.
-	[Intel® Gaudi®  Tutorials](https://github.com/HabanaAI/Gaudi-tutorials): Tutorials with step-by-step instructions for running PyTorch and PyTorch Lightning models on the Intel Gaudi AI Processor for training and inferencing, from beginner level to advanced users.
-	[Powered-by-Intel Leaderboard](https://huggingface.co/spaces/Intel/powered_by_intel_llm_leaderboard): This leaderboard celebrates and increases the discoverability of models developed on Intel hardware by the AI developer community. We provide developers with sample code and resources (developer programs) to deploy (inference) AI PC, Intel® Xeon® Scalable processors, Intel® Gaudi® processors, Intel® Arc™ GPUs, and Intel® Data Center GPUs.
-	[Intel® AI Reference Models](https://github.com/intel/models): This repository contains links to pre-trained models, sample scripts, best practices, and step-by-step tutorials for many popular open-source machine learning models optimized by Intel to run on Intel® Xeon® Scalable processors and Intel® Data Center GPUs.
-	[awesome-oneapi](https://github.com/oneapi-community/awesome-oneapi): A community sourced list of awesome oneAPI and SYCL projects for solutions across a wide range of industry segments.
- [Generative AI Examples](https://github.com/opea-project/GenAIExamples): A collection of GenAI examples such as ChatQnA, Copilot, which illustrate the pipeline capabilities of the Open Platform for Enterprise AI (OPEA) project. OPEA is an ecosystem orchestration framework to integrate performant GenAI technologies & workflows leading to quicker GenAI adoption and business value.

## Licenses

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/main/License.txt) for details.

Third-party program licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/main/third-party-programs.txt).

## Notices and Disclaimers

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.
